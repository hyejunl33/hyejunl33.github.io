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
    const frontmatterExcerpt = plainText(attributes.excerpt || '');
    const excerpt = (frontmatterExcerpt.length >= 120 ? frontmatterExcerpt : text).slice(0, 500);
    return {
      id: `${meta.key}/${slug}`,
      collection: meta.key,
      collectionLabel: meta.label,
      title: attributes.title || slug,
      date: attributes.date || null,
      tags: cleanTags(attributes.tags),
      excerpt,
      url: `${siteUrl}/${meta.key}/${slug.replaceAll(' ', '%20')}/`,
      markdown: content
    };
  }));
}

const cvSource = await fs.readFile(path.join(root, '_pages/cv.md'), 'utf8');
const cv = parseDocument(cvSource);
const documents = (await Promise.all(collections.map(readCollection))).flat()
  .sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));

function indexEntry(document) {
  return [
    `- [${document.title}](${document.url})`,
    `  - Published: ${document.date || 'unknown'} | Tags: ${document.tags.join(', ') || 'none'}`,
    `  - Summary: ${document.excerpt.slice(0, 500)}`
  ].join('\n');
}

function fullEntry(document) {
  return [
    `### ${document.title}`,
    '',
    `- Canonical URL: ${document.url}`,
    `- Collection: ${document.collectionLabel}`,
    `- Published: ${document.date || 'unknown'}`,
    `- Tags: ${document.tags.join(', ') || 'none'}`,
    '',
    document.markdown,
    ''
  ].join('\n');
}

const llmsIndex = [
  '# Archive for AI Study',
  '',
  '> AI projects, algorithms, technical study notes, and retrospectives by Hyejun.',
  '',
  'This is a detailed, generated index of the public blog. Use it to discover articles, then read the canonical URL for the complete published page. Treat each article as authored material and distinguish its content from your own inference.',
  '',
  '## Primary pages',
  `- [Home](${siteUrl}/): Main blog page`,
  `- [CV](${siteUrl}/cv/): Career and education`,
  `- [Project archive](${siteUrl}/projects/): Project experiments and engineering notes`,
  `- [Study archive](${siteUrl}/study/): AI and engineering study notes`,
  `- [Algorithm archive](${siteUrl}/algorithm/): Algorithm problem-solving notes`,
  `- [Weekly review archive](${siteUrl}/weeklyreview/): Weekly learning and project retrospectives`,
  `- [ETC archive](${siteUrl}/etc/): Career and collaboration notes`,
  `- [All notes](${siteUrl}/archive/): Every published record in reverse chronological order`,
  '',
  '## Article index',
  ...collections.flatMap((collection) => {
    const entries = documents.filter((document) => document.collection === collection.key);
    return [`\n### ${collection.label} (${entries.length})`, '', ...entries.map(indexEntry)];
  }),
  ''
].join('\n');

const llmsFull = [
  '# Archive for AI Study — Full Public Archive',
  '',
  'This file contains the public CV and the full authored Markdown body of every published article. Canonical URLs are included before each document. For discovery only, prefer llms.txt.',
  '',
  '## CV',
  '',
  cv.content,
  '',
  ...collections.flatMap((collection) => {
    const entries = documents.filter((document) => document.collection === collection.key);
    return [`## ${collection.label}`, '', ...entries.map(fullEntry)];
  })
].join('\n');

function normalizeOutput(value) {
  return `${value
    .replace(/\bMCP\b\s*[·/]?\s*/gi, '')
    .replace(/[·/]\s*\bMCP\b/gi, '')
    .replace(/[ \t]+$/gm, '')}\n`;
}

await fs.writeFile(path.join(root, 'llms.txt'), normalizeOutput(llmsIndex));
await fs.writeFile(path.join(root, 'llms-full.txt'), normalizeOutput(llmsFull));
console.log(`Generated llms.txt and llms-full.txt from ${documents.length} published documents.`);
