# ssew-proxy

A Cloudflare Worker in front of the scoring API on AWS. The demo page
(https://rejusamjohn.pages.dev/early-warning/) calls it from the browser, and learners can call it
from `curl` or Python. It checks each request, applies rate limits and forwards what passes to
the API. It stores nothing.

```
browser, curl or Python ──▶ ssew-proxy (Cloudflare) ──▶ API Gateway + Lambda (AWS)
```

## Routes

| Route | What it does |
|---|---|
| `GET /` | Plain-text help. |
| `GET /health` | Forwards the API's `/health`: status and model version for each week. |
| `POST /v1/score` | Checks the body, then forwards it to the API's `/v1/score`. The API's status and body come back unchanged, including its 422 validation details. |

Errors from the proxy itself are JSON: `{"error": "<plain sentence>"}`.

## Checks and limits

All numbers are in [`limits.json`](limits.json); `test/config.test.ts` checks that
`wrangler.jsonc` uses the same ones.

- Body: `Content-Type: application/json`, at most 8192 bytes, a JSON object with `week` (2, 4,
  8 or 12) and `students` (1 to 5 objects with exactly the API's 15 fields, numbers where the
  API wants numbers, a known module code). The full rules stay in the API; these checks stop
  oversized or malformed traffic before it reaches AWS.
- Per address: 60 calls a minute, then 429 with `Retry-After: 60`.
- All visitors together: 240 calls a minute to AWS, below the API throttle of 5 a second, then
  503. Cloudflare documents rate-limit bindings as approximate and per location, so the AWS
  throttle and the AWS budget alarm remain the backstop.
- AWS slow or down: 504 after 29 seconds, or 502, both as JSON.

CORS lets browsers call it from `https://rejusamjohn.pages.dev` and `http://localhost:5500`.
Requests with no `Origin` header (curl, Python) are not browser requests and are served as
normal.

There is no edge cache: the Workers Cache API has no effect on `*.workers.dev` hosts
(https://developers.cloudflare.com/r2/examples/cache-api/). The page instead reads the advisor
lists from files scored once in a batch, and remembers answers it already has during a visit.

No secrets: the API address is a plain variable in `wrangler.jsonc`. Invocation logs are off,
so client IP addresses are not stored.

## Try it

```bash
curl -s https://ssew-proxy.rejusamjohn.workers.dev/health

python3 -c 'import json; r = json.load(open("docs/examples/score_request.json")); r["students"] = r["students"][:1]; print(json.dumps(r))' > one.json
curl -s -X POST https://ssew-proxy.rejusamjohn.workers.dev/v1/score \
  -H 'Content-Type: application/json' -d @one.json
```

## Develop and deploy

```bash
cd proxy
npm ci
npm run check   # TypeScript
npm test        # vitest
npx wrangler login    # once
npx wrangler deploy
```

CI runs the type check and tests on pull requests and pushes to main. It does not deploy; there is no Cloudflare token
in GitHub.
