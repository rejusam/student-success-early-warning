// ssew-proxy: a public front door to the student early-warning scoring API on AWS.
// It checks each request, applies rate limits and forwards what passes. It stores nothing.
import { checkScoreBody } from './check';
import { corsHeaders } from './cors';
import limits from '../limits.json';

export interface Env {
  IP_LIMITER: RateLimit;
  UPSTREAM_LIMITER: RateLimit;
  AWS_API_URL: string;
}

const REPO = 'https://github.com/rejusam/student-success-early-warning';
const UPSTREAM_TIMEOUT_MS = 29000; // just under the API Gateway limit of 30 seconds

const HELP = `Student early-warning scoring API (proxy)

A worked example of deploying a model. Scores come from a logistic regression trained on the
public Open University Learning Analytics Dataset (OULAD); risk is the probability of ending
a module with Fail or Withdrawn.

Routes
  GET  /health     status and model versions
  POST /v1/score   {"week": 2|4|8|12, "students": [ ...1 to ${limits.max_students} students... ]}

Limits: ${limits.max_students} students and ${limits.max_body_bytes} bytes per call, ${limits.ip_per_minute} calls per minute per address.

Example
  curl -s -X POST https://ssew-proxy.rejusamjohn.workers.dev/v1/score \\
    -H 'Content-Type: application/json' \\
    -d @score_request.json

Code and a full example request: ${REPO}
`;

function json(status: number, body: unknown, cors: Record<string, string>, extra: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', 'cache-control': 'no-store', ...cors, ...extra },
  });
}

const error = (status: number, message: string, cors: Record<string, string>, extra?: Record<string, string>) =>
  json(status, { error: message }, cors, extra);

async function forward(url: string, init: RequestInit, cors: Record<string, string>): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(url, { ...init, signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'TimeoutError') {
      return error(504, 'The scoring API did not answer in time. Try again in a minute.', cors);
    }
    return error(502, 'The scoring API could not be reached.', cors);
  }
  const text = await upstream.text();
  try {
    return json(upstream.status, JSON.parse(text), cors);
  } catch {
    return error(502, `The scoring API answered ${upstream.status} without JSON.`, cors);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const cors = corsHeaders(request.headers.get('origin'));

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          ...cors,
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    if (url.pathname === '/' && request.method === 'GET') {
      return new Response(HELP, { headers: { 'content-type': 'text/plain; charset=utf-8', ...cors } });
    }

    const routes: Record<string, string> = { '/health': 'GET', '/v1/score': 'POST' };
    const method = routes[url.pathname];
    if (!method) return error(404, 'Not found. The routes are GET /health and POST /v1/score.', cors);
    if (request.method !== method) {
      return error(405, `Use ${method} for ${url.pathname}.`, cors, { Allow: `${method}, OPTIONS` });
    }

    // Cheap checks first, so a bad request does not use up anyone's limit.
    let raw = '';
    if (method === 'POST') {
      const type = request.headers.get('content-type') ?? '';
      if (!type.toLowerCase().startsWith('application/json')) {
        return error(415, 'Send the body as JSON with Content-Type: application/json.', cors);
      }
      const length = Number(request.headers.get('content-length') ?? 0);
      if (length > limits.max_body_bytes) {
        return error(413, `The body is larger than ${limits.max_body_bytes} bytes.`, cors);
      }
      raw = await request.text();
      const checked = checkScoreBody(raw);
      if (!checked.ok) return error(checked.status, checked.error, cors);
    }

    const ip = request.headers.get('cf-connecting-ip') ?? 'unknown';
    if (!(await env.IP_LIMITER.limit({ key: ip })).success) {
      return error(429, 'Too many calls from this address. Wait a minute and try again.', cors, {
        'Retry-After': '60',
      });
    }
    if (!(await env.UPSTREAM_LIMITER.limit({ key: 'aws' })).success) {
      return error(503, 'The service is busy. Try again in a minute.', cors, { 'Retry-After': '60' });
    }

    const target = env.AWS_API_URL + url.pathname;
    if (method === 'GET') return forward(target, { method: 'GET' }, cors);
    return forward(target, { method: 'POST', body: raw, headers: { 'content-type': 'application/json' } }, cors);
  },
};
