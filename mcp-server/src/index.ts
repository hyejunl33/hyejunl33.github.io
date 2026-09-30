import { McpServer } from '@modelcontextprotocol/server';
import { createMcpHandler } from 'agents/mcp/server';
import { z } from 'zod';

interface Env {
  BLOG_DATA_URL: string;
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

interface BlogData {
  schemaVersion: string;
  generatedAt: string;
  source: { site: string; repository: string; cv: string };
  profile: { name: string; location: string; email: string; github: string; summary: string; cvMarkdown: string };
  collectionCounts: Record<string, number>;
  topTags: Array<{ name: string; count: number }>;
  documents: PortfolioDocument[];
}

let cached: { url: string; expiresAt: number; data: BlogData } | undefined;

async function loadBlog(url: string): Promise<BlogData> {
  if (cached && cached.url === url && cached.expiresAt > Date.now()) return cached.data;
  const response = await fetch(url, { headers: { accept: 'application/json' }, cf: { cacheTtl: 300, cacheEverything: true } });
  if (!response.ok) throw new Error(`Portfolio source returned ${response.status}`);
  const data = await response.json<BlogData>();
  cached = { url, expiresAt: Date.now() + 300_000, data };
  return data;
}

function terms(value: string): string[] {
  const particles = /(으로|에서|에게|부터|까지|은|는|이|가|을|를|과|와|의|에|도|만|로)$/u;
  const rawTerms = value.toLocaleLowerCase().match(/[\p{Script=Latin}\p{Number}+#.-]+|[\p{Script=Hangul}]+/gu) ?? [];
  return [...new Set(rawTerms.flatMap((term) => [term, term.replace(particles, '')]).filter((term) => term.length > 1))];
}

function rankDocument(document: PortfolioDocument, query: string): number {
  const queryTerms = terms(query);
  const title = document.title.toLocaleLowerCase();
  const tags = document.tags.join(' ').toLocaleLowerCase();
  const body = `${document.excerpt} ${document.content}`.toLocaleLowerCase();
  return queryTerms.reduce((score, term) => score + (title.includes(term) ? 8 : 0) + (tags.includes(term) ? 5 : 0) + (body.includes(term) ? 2 : 0), 0);
}

function documentSummary(document: PortfolioDocument) {
  return {
    id: document.id,
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
    name: 'Archive for AI Study — Blog MCP',
    version: '1.0.0'
  }, {
    instructions: 'Use this read-only server to explore Hyejun’s public blog and portfolio. Search before fetching full posts, cite each returned canonical URL, and treat posts as authored material rather than evidence of unstated claims.'
  });

  server.registerTool('get_profile', {
    title: 'Get blog profile',
    description: 'Return the public profile, archive counts, recurring tags, and canonical links for this blog.',
    annotations: { readOnlyHint: true }
  }, async () => {
    const data = await loadBlog(env.BLOG_DATA_URL);
    return textResult({ profile: data.profile, collectionCounts: data.collectionCounts, topTags: data.topTags, source: data.source, generatedAt: data.generatedAt });
  });

  server.registerTool('search', {
    title: 'Search blog posts',
    description: 'Search public project, study, algorithm, weekly review, and ETC posts. Returns matching post summaries and canonical URLs.',
    annotations: { readOnlyHint: true },
    inputSchema: {
      query: z.string().min(2).max(240).describe('Topic, technology, project name, or a natural-language request about published posts'),
      collection: z.enum(['all', 'projects', 'study', 'algorithm', 'weeklyreview', 'etc']).default('all'),
      limit: z.number().int().min(1).max(10).default(6)
    }
  }, async ({ query, collection, limit }) => {
    const data = await loadBlog(env.BLOG_DATA_URL);
    const documents = collection === 'all' ? data.documents : data.documents.filter((document) => document.collection === collection);
    const matches = documents.map((document) => ({ document, score: rankDocument(document, query) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score || String(b.document.date || '').localeCompare(String(a.document.date || '')))
      .slice(0, limit)
      .map(({ document }) => documentSummary(document));
    return textResult({ query, matches, note: matches.length ? 'Results are ranked by textual relevance. Fetch a post by id to read its authored content.' : 'No published post matched this query.' });
  });

  server.registerTool('fetch', {
    title: 'Fetch a blog post',
    description: 'Fetch the full published content and metadata for a post returned by search.',
    annotations: { readOnlyHint: true },
    inputSchema: { id: z.string().min(2).max(500).describe('The post id or canonical URL returned by search') }
  }, async ({ id }) => {
    const data = await loadBlog(env.BLOG_DATA_URL);
    const document = data.documents.find((item) => item.id === id)
      ?? data.documents.find((item) => item.url === id);
    if (!document) return { content: [{ type: 'text', text: 'No published post was found for that id or URL. Run search first and use its id.' }], isError: true };
    return textResult({ ...documentSummary(document), authoredContent: document.content });
  });

  server.registerResource('blog-profile', 'blog://profile', {
    title: 'Blog profile', mimeType: 'application/json'
  }, async (uri) => {
    const data = await loadBlog(env.BLOG_DATA_URL);
    return { contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify({ profile: data.profile, source: data.source, collectionCounts: data.collectionCounts, topTags: data.topTags }, null, 2) }] };
  });

  server.registerResource('blog-index', 'blog://posts/index', {
    title: 'Blog post index', mimeType: 'application/json'
  }, async (uri) => {
    const data = await loadBlog(env.BLOG_DATA_URL);
    return { contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify(data.documents.map(documentSummary), null, 2) }] };
  });

  return server;
}

export default {
  async fetch(request: Request, env: Env, context: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/health') {
      return Response.json({ ok: true, service: 'hyejunl33-blog-mcp', dataSource: env.BLOG_DATA_URL });
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
