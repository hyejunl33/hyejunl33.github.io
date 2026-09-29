import { McpServer } from '@modelcontextprotocol/server';
import { createMcpHandler } from 'agents/mcp/server';
import { z } from 'zod';

interface Env {
  PORTFOLIO_DATA_URL: string;
}

interface PortfolioDocument {
  id: string;
  collection: string;
  collectionLabel: string;
  title: string;
  date: string | null;
  tags: string[];
  excerpt: string;
  url: string;
  content: string;
}

interface PortfolioData {
  schemaVersion: string;
  generatedAt: string;
  source: { site: string; repository: string; cv: string };
  candidate: { name: string; location: string; email: string; github: string; summary: string; cvMarkdown: string };
  collectionCounts: Record<string, number>;
  topTags: Array<{ name: string; count: number }>;
  documents: PortfolioDocument[];
}

let cached: { url: string; expiresAt: number; data: PortfolioData } | undefined;

async function loadPortfolio(url: string): Promise<PortfolioData> {
  if (cached && cached.url === url && cached.expiresAt > Date.now()) return cached.data;
  const response = await fetch(url, { headers: { accept: 'application/json' }, cf: { cacheTtl: 300, cacheEverything: true } });
  if (!response.ok) throw new Error(`Portfolio source returned ${response.status}`);
  const data = await response.json<PortfolioData>();
  cached = { url, expiresAt: Date.now() + 300_000, data };
  return data;
}

function terms(value: string): string[] {
  return value.toLocaleLowerCase().split(/[^\p{Letter}\p{Number}+#.-]+/u).filter((term) => term.length > 1);
}

function rankDocument(document: PortfolioDocument, query: string): number {
  const queryTerms = terms(query);
  const title = document.title.toLocaleLowerCase();
  const tags = document.tags.join(' ').toLocaleLowerCase();
  const body = `${document.excerpt} ${document.content}`.toLocaleLowerCase();
  return queryTerms.reduce((score, term) => score + (title.includes(term) ? 8 : 0) + (tags.includes(term) ? 5 : 0) + (body.includes(term) ? 2 : 0), 0);
}

function evidence(document: PortfolioDocument) {
  return {
    title: document.title,
    collection: document.collectionLabel,
    date: document.date,
    tags: document.tags,
    excerpt: document.excerpt,
    url: document.url
  };
}

function textResult(value: unknown) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }] };
}

function createServer(env: Env) {
  const server = new McpServer({
    name: 'Archive for AI Study — Recruiter Portfolio',
    version: '1.0.0'
  }, {
    instructions: 'Use this read-only server to inspect Hyejun’s public portfolio. Cite returned source URLs, separate facts from inference, and do not infer employment claims that are not present in the evidence.'
  });

  server.registerTool('get_candidate_snapshot', {
    title: 'Candidate snapshot',
    description: 'Return the public CV, archive counts, recurring technologies, and canonical profile links for a fast recruiter briefing.'
  }, async () => {
    const data = await loadPortfolio(env.PORTFOLIO_DATA_URL);
    return textResult({ candidate: data.candidate, collectionCounts: data.collectionCounts, topTags: data.topTags, source: data.source, generatedAt: data.generatedAt });
  });

  server.registerTool('search_portfolio', {
    title: 'Search portfolio evidence',
    description: 'Search authored project, algorithm, review, and ETC posts. Returns evidence snippets and original URLs; use the URLs when making recruiter-facing claims.',
    inputSchema: {
      query: z.string().min(2).max(240).describe('Skills, role requirements, project topics, or a natural-language evidence question'),
      collection: z.enum(['all', 'projects', 'study', 'algorithm', 'weeklyreview', 'etc']).default('all'),
      limit: z.number().int().min(1).max(10).default(6)
    }
  }, async ({ query, collection, limit }) => {
    const data = await loadPortfolio(env.PORTFOLIO_DATA_URL);
    const candidates = collection === 'all' ? data.documents : data.documents.filter((document) => document.collection === collection);
    const matches = candidates.map((document) => ({ document, score: rankDocument(document, query) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score || String(b.document.date || '').localeCompare(String(a.document.date || '')))
      .slice(0, limit)
      .map(({ document, score }) => ({ score, ...evidence(document) }));
    return textResult({ query, matches, note: matches.length ? 'Scores rank textual relevance only; they are not candidate evaluation scores.' : 'No direct authored evidence matched this query.' });
  });

  server.registerTool('get_project_case_study', {
    title: 'Read a project case study',
    description: 'Return the closest matching project note with a longer authored excerpt and its canonical URL.',
    inputSchema: { titleOrTopic: z.string().min(2).max(240) }
  }, async ({ titleOrTopic }) => {
    const data = await loadPortfolio(env.PORTFOLIO_DATA_URL);
    const match = data.documents.filter((document) => document.collection === 'projects')
      .map((document) => ({ document, score: rankDocument(document, titleOrTopic) }))
      .sort((a, b) => b.score - a.score)[0];
    if (!match || match.score === 0) return { content: [{ type: 'text', text: 'No matching project case study was found.' }], isError: true };
    return textResult({ ...evidence(match.document), authoredContent: match.document.content.slice(0, 8000) });
  });

  server.registerTool('get_role_evidence', {
    title: 'Gather evidence for a role',
    description: 'Gather authored evidence relevant to a job description without producing a hiring score or unsupported claim.',
    inputSchema: { roleOrRequirements: z.string().min(10).max(2000), limit: z.number().int().min(3).max(12).default(8) }
  }, async ({ roleOrRequirements, limit }) => {
    const data = await loadPortfolio(env.PORTFOLIO_DATA_URL);
    const matches = data.documents.map((document) => ({ document, score: rankDocument(document, roleOrRequirements) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(({ document }) => evidence(document));
    return textResult({ roleOrRequirements, evidence: matches, guidance: 'Assess fit using only the linked authored evidence. Explicitly identify missing evidence and keep any conclusion separate from facts.' });
  });

  server.registerResource('candidate-profile', 'portfolio://candidate/profile', {
    title: 'Candidate profile', mimeType: 'application/json'
  }, async (uri) => {
    const data = await loadPortfolio(env.PORTFOLIO_DATA_URL);
    return { contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify({ candidate: data.candidate, source: data.source, collectionCounts: data.collectionCounts, topTags: data.topTags }, null, 2) }] };
  });

  server.registerResource('portfolio-index', 'portfolio://evidence/index', {
    title: 'Portfolio evidence index', mimeType: 'application/json'
  }, async (uri) => {
    const data = await loadPortfolio(env.PORTFOLIO_DATA_URL);
    return { contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify(data.documents.map(evidence), null, 2) }] };
  });

  server.registerPrompt('evaluate_candidate_with_evidence', {
    title: 'Evidence-based candidate review',
    description: 'Prepare a recruiter review grounded in this portfolio and a role description.',
    argsSchema: { role: z.string().describe('Role title or job description') }
  }, ({ role }) => ({
    messages: [{
      role: 'user',
      content: {
        type: 'text',
        text: `Review Hyejun's public portfolio for the following role: ${role}\n\nFirst call get_candidate_snapshot and get_role_evidence. Then write: (1) verified strengths with source URLs, (2) relevant project evidence, (3) missing or unclear evidence, and (4) focused interview questions. Never fabricate a skill or produce a numeric hiring score.`
      }
    }]
  }));

  return server;
}

export default {
  async fetch(request: Request, env: Env, context: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/health') {
      return Response.json({ ok: true, service: 'hyejunl33-portfolio-mcp', dataSource: env.PORTFOLIO_DATA_URL });
    }
    if (url.pathname !== '/mcp') return new Response('Not found', { status: 404 });

    const origin = request.headers.get('origin');
    const originUrl = origin ? new URL(origin) : null;
    const isLoopback = originUrl && originUrl.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(originUrl.hostname);
    const isPortfolio = originUrl && originUrl.origin === 'https://hyejunl33.github.io';
    if (origin && !isLoopback && !isPortfolio) return new Response('Origin not allowed', { status: 403 });

    return createMcpHandler(() => createServer(env))(request, env, context);
  }
} satisfies ExportedHandler<Env>;
