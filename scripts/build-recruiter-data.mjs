import { promises as fs } from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const siteUrl = 'https://hyejunl33.github.io';
const collections = [
  { key: 'projects', directory: '_projects', label: 'Project' },
  { key: 'study', directory: '_study', label: 'Study' },
  { key: 'algorithm', directory: '_algorithm', label: 'Algorithm' },
  { key: 'weeklyreview', directory: '_weeklyreview', label: 'WeeklyReview' },
  { key: 'etc', directory: '_etc', label: 'ETC' }
];

function unquote(value = '') {
  return value.trim().replace(/^&\S+\s+/, '').replace(/^['"]|['"]$/g, '');
}

function parseDocument(source) {
  const match = source.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/);
  if (!match) return { attributes: {}, content: source };
  const lines = match[1].split(/\r?\n/);
  const attributes = {};
  let listKey = null;

  for (const line of lines) {
    const listItem = line.match(/^\s+-\s+(.+)$/);
    if (listItem && listKey) {
      attributes[listKey].push(unquote(listItem[1]).replace(/^\[|\]$/g, ''));
      continue;
    }
    const pair = line.match(/^([\w-]+):\s*(.*)$/);
    if (!pair) continue;
    const [, key, rawValue] = pair;
    if (!rawValue) {
      attributes[key] = [];
      listKey = key;
      continue;
    }
    listKey = null;
    if (rawValue === 'true' || rawValue === 'false') attributes[key] = rawValue === 'true';
    else if (rawValue.startsWith('[') && rawValue.endsWith(']')) attributes[key] = rawValue.slice(1, -1).split(',').map(unquote);
    else attributes[key] = unquote(rawValue);
  }
  return { attributes, content: match[2].trim() };
}

function plainText(markdown) {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[`*_>#|~=-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function cleanTags(tags) {
  if (!Array.isArray(tags)) return [];
  return [...new Set(tags.flatMap((tag) => String(tag).replace(/^\[|\]$/g, '').split(',')).map((tag) => tag.trim()).filter(Boolean))];
}

async function readCollection(meta) {
  const directory = path.join(root, meta.directory);
  const names = (await fs.readdir(directory)).filter((name) => name.endsWith('.md')).sort();
  return Promise.all(names.map(async (name) => {
    const source = await fs.readFile(path.join(directory, name), 'utf8');
    const { attributes, content } = parseDocument(source);
    const slug = name.replace(/\.md$/, '');
    const text = plainText(content);
    const excerpt = plainText(attributes.excerpt || '').slice(0, 420) || text.slice(0, 420);
    return {
      id: `${meta.key}/${slug}`,
      collection: meta.key,
      collectionLabel: meta.label,
      title: attributes.title || slug,
      date: attributes.date || null,
      tags: cleanTags(attributes.tags),
      excerpt,
      url: `${siteUrl}/${meta.key}/${slug.replaceAll(' ', '%20')}/`,
      content: text.slice(0, 12000)
    };
  }));
}

const config = await fs.readFile(path.join(root, '_config.yml'), 'utf8');
const cvSource = await fs.readFile(path.join(root, '_pages/cv.md'), 'utf8');
const cv = parseDocument(cvSource);
const documents = (await Promise.all(collections.map(readCollection))).flat()
  .sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));
const tagCounts = new Map();
documents.forEach((document) => document.tags.forEach((tag) => tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1)));
const topTags = [...tagCounts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 24)
  .map(([name, count]) => ({ name, count }));
const getConfigValue = (key) => unquote((config.match(new RegExp(`^${key}\\s*:\\s*(.+)$`, 'm')) || [])[1] || '');
const getAuthorValue = (key) => unquote((config.match(new RegExp(`^\\s{2}${key}\\s*:\\s*(.+)$`, 'm')) || [])[1] || '');

const payload = {
  schemaVersion: '1.0.0',
  generatedAt: new Date().toISOString(),
  source: {
    site: siteUrl,
    repository: 'https://github.com/hyejunl33/hyejunl33.github.io',
    cv: `${siteUrl}/cv/`
  },
  candidate: {
    name: getAuthorValue('name') || 'Hyejun',
    location: getAuthorValue('location'),
    email: getAuthorValue('email'),
    github: `https://github.com/${getAuthorValue('github') || 'hyejunl33'}`,
    summary: getConfigValue('description'),
    cvMarkdown: cv.content
  },
  collectionCounts: Object.fromEntries(collections.map(({ key }) => [key, documents.filter((document) => document.collection === key).length])),
  topTags,
  documents
};

await fs.mkdir(path.join(root, 'assets/data'), { recursive: true });
await fs.writeFile(path.join(root, 'assets/data/recruiter-portfolio.json'), `${JSON.stringify(payload, null, 2)}\n`);

const llmsIndex = [
  '# Archive for AI Study',
  '',
  `> ${payload.candidate.summary}`,
  '',
  'This is the public AI/engineering portfolio of Hyejun. Prefer the original URLs below as evidence and distinguish authored claims from your own inference.',
  '',
  '## Primary pages',
  `- [CV](${payload.source.cv}): Career and education`,
  `- [Project archive](${siteUrl}/projects/): Project experiments and engineering notes`,
  `- [Study archive](${siteUrl}/study/): All learning records`,
  `- [Recruiter guide](${siteUrl}/recruiter/): MCP connection and evidence explorer`,
  `- [Structured portfolio data](${siteUrl}/assets/data/recruiter-portfolio.json): Machine-readable public evidence`,
  '',
  '## Recent portfolio evidence',
  ...documents.slice(0, 16).map((document) => `- [${document.title}](${document.url}): ${document.excerpt.slice(0, 180)}`),
  ''
].join('\n');

const llmsFull = [
  llmsIndex,
  '## CV',
  cv.content,
  '',
  '## Portfolio documents',
  ...documents.map((document) => `\n### ${document.title}\n\n- Collection: ${document.collectionLabel}\n- Date: ${document.date || 'unknown'}\n- URL: ${document.url}\n- Tags: ${document.tags.join(', ') || 'none'}\n\n${document.content}\n`)
].join('\n');

await fs.writeFile(path.join(root, 'llms.txt'), `${llmsIndex}\n`);
await fs.writeFile(path.join(root, 'llms-full.txt'), `${llmsFull}\n`);
console.log(`Generated recruiter dataset with ${documents.length} documents.`);
