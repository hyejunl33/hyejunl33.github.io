# Archive for AI Study — Recruiter MCP

Public, read-only Streamable HTTP MCP server for recruiter-facing portfolio discovery.

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

After Cloudflare returns the final `https://*.workers.dev/mcp` URL, set `_config.yml` → `mcp_endpoint` to that URL and rebuild the Jekyll site. The server reads only the public generated dataset at `assets/data/recruiter-portfolio.json`; it has no write tools and stores no recruiter queries.
