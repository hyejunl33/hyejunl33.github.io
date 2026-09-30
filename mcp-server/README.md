# Archive for AI Study — Blog MCP

Public, read-only Streamable HTTP MCP server for searching and reading the published Archive for AI Study blog and portfolio.

## Available tools

- `search`: Find published posts by topic, collection, or natural-language query.
- `fetch`: Read one post returned by `search`, using its id or canonical URL.
- `get_profile`: Read the public profile and archive metadata.

Every post result includes its canonical URL. The server does not evaluate people, create scores, or store queries.

## Example requests

- “이 블로그에서 Airflow를 실제로 구현한 글을 찾아서 원문 링크와 함께 보여줘.”
- “이 포트폴리오에 있는 멀티에이전트 관련 글을 전부 찾아줘.”
- “이 블로그의 프로젝트 글만 가져와서 읽어줘.”

## Local verification

```bash
npm install
npm run check
npm run dev
```

The MCP endpoint is `http://localhost:8787/mcp` (use the port printed by Wrangler). Test it with `npx @modelcontextprotocol/inspector@latest`.

## Deploy

```bash
npm run deploy
```

After Cloudflare returns the final `https://*.workers.dev/mcp` URL, set `_config.yml` → `mcp_endpoint` to that URL and rebuild the Jekyll site. The server reads only the public generated dataset at `assets/data/blog-content.json`; it has no write tools and stores no queries.
