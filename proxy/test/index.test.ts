import { afterEach, describe, expect, it, vi } from 'vitest';
import example from '../../docs/examples/score_request.json';
import worker, { type Env } from '../src/index';

const AWS = 'https://aws.example';
const valid = JSON.stringify({ week: 8, students: [example.students[0]] });

function env(ip = true, upstream = true): Env {
  return {
    AWS_API_URL: AWS,
    IP_LIMITER: { limit: async () => ({ success: ip }) },
    UPSTREAM_LIMITER: { limit: async () => ({ success: upstream }) },
  } as unknown as Env;
}

function upstream(response: () => Response | Promise<Response>) {
  const fake = vi.fn(async () => response());
  vi.stubGlobal('fetch', fake);
  return fake;
}

function post(body: string, headers: Record<string, string> = {}) {
  return new Request('https://proxy.example/v1/score', {
    method: 'POST',
    body,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe('routes', () => {
  it('GET / explains the API in plain text', async () => {
    const res = await worker.fetch(new Request('https://proxy.example/'), env());
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/plain');
    const text = await res.text();
    expect(text).toContain('curl');
    expect(text).toContain('/v1/score');
  });

  it('GET /health forwards the API health', async () => {
    const fake = upstream(() => Response.json({ status: 'ok' }));
    const res = await worker.fetch(new Request('https://proxy.example/health'), env());
    expect(fake).toHaveBeenCalledWith(`${AWS}/health`, expect.anything());
    expect(await res.json()).toEqual({ status: 'ok' });
  });

  it('POST /v1/score forwards a valid body once and passes the answer back', async () => {
    const fake = upstream(() => Response.json({ week: 8, results: [] }, { status: 200 }));
    const res = await worker.fetch(post(valid), env());
    expect(fake).toHaveBeenCalledTimes(1);
    const [url, init] = fake.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${AWS}/v1/score`);
    expect(init.method).toBe('POST');
    expect(init.body).toBe(valid);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/json');
    expect(res.headers.get('cache-control')).toBe('no-store');
    expect(await res.json()).toEqual({ week: 8, results: [] });
  });

  it('passes an API 422 back with its body', async () => {
    upstream(() => Response.json({ detail: [{ msg: 'bad' }] }, { status: 422 }));
    const res = await worker.fetch(post(valid), env());
    expect(res.status).toBe(422);
    expect(await res.json()).toEqual({ detail: [{ msg: 'bad' }] });
  });

  it('GET /v1/score is 405', async () => {
    const res = await worker.fetch(new Request('https://proxy.example/v1/score'), env());
    expect(res.status).toBe(405);
    expect(res.headers.get('allow')).toBe('POST, OPTIONS');
  });

  it('an unknown path is a 404 JSON error', async () => {
    const res = await worker.fetch(new Request('https://proxy.example/nope'), env());
    expect(res.status).toBe(404);
    expect(((await res.json()) as { error: string }).error).toBeTruthy();
  });
});

describe('checks before AWS is called', () => {
  it('rejects a non-JSON content type with 415', async () => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(post(valid, { 'content-type': 'text/plain' }), env());
    expect(res.status).toBe(415);
    expect(fake).not.toHaveBeenCalled();
  });

  it('rejects a huge Content-Length with 413 before reading the body', async () => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(post(valid, { 'content-length': '999999' }), env());
    expect(res.status).toBe(413);
    expect(fake).not.toHaveBeenCalled();
  });

  it.each(['', 'null', '[]'])('rejects body %j with 400 JSON', async (raw) => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(post(raw), env());
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toBeTruthy();
    expect(fake).not.toHaveBeenCalled();
  });

  it('rejects a body over the limit when Content-Length is missing or wrong', async () => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(post(' '.repeat(9000) + valid), env());
    expect(res.status).toBe(413);
    expect(fake).not.toHaveBeenCalled();
  });
});

describe('streamed body', () => {
  it('stops reading a chunked body once it passes the limit', async () => {
    const fake = upstream(() => Response.json({}));
    let pulled = 0;
    const chunk = new TextEncoder().encode(' '.repeat(1024));
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled += 1;
        if (pulled > 1000) controller.close();
        else controller.enqueue(chunk);
      },
    });
    const request = new Request('https://proxy.example/v1/score', {
      method: 'POST',
      body: stream,
      headers: { 'content-type': 'application/json' },
      // @ts-expect-error duplex is needed for a stream body in Node's fetch
      duplex: 'half',
    });
    const res = await worker.fetch(request, env());
    expect(res.status).toBe(413);
    expect(pulled).toBeLessThan(20);
    expect(fake).not.toHaveBeenCalled();
  });
});

describe('rate limits', () => {
  it('per-IP limit gives 429 with Retry-After and no AWS call', async () => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(post(valid), env(false, true));
    expect(res.status).toBe(429);
    expect(res.headers.get('retry-after')).toBe('60');
    expect(fake).not.toHaveBeenCalled();
  });

  it('upstream limit gives 503 and no AWS call', async () => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(post(valid), env(true, false));
    expect(res.status).toBe(503);
    expect(fake).not.toHaveBeenCalled();
  });

  it('applies the limits to /health too', async () => {
    const fake = upstream(() => Response.json({}));
    const res = await worker.fetch(new Request('https://proxy.example/health'), env(false, true));
    expect(res.status).toBe(429);
    expect(fake).not.toHaveBeenCalled();
  });
});

describe('upstream failures', () => {
  it('a timeout gives 504 JSON', async () => {
    upstream(() => {
      throw new DOMException('timed out', 'TimeoutError');
    });
    const res = await worker.fetch(post(valid), env());
    expect(res.status).toBe(504);
    expect(((await res.json()) as { error: string }).error).toContain('did not answer');
  });

  it('a network error gives 502 JSON', async () => {
    upstream(() => {
      throw new TypeError('fetch failed');
    });
    const res = await worker.fetch(post(valid), env());
    expect(res.status).toBe(502);
    expect(((await res.json()) as { error: string }).error).toBeTruthy();
  });

  it('a 502 with a non-JSON body gives 502 JSON', async () => {
    upstream(() => new Response('<html>Bad gateway</html>', { status: 502 }));
    const res = await worker.fetch(post(valid), env());
    expect(res.status).toBe(502);
    expect(res.headers.get('content-type')).toBe('application/json');
    expect(((await res.json()) as { error: string }).error).toBeTruthy();
  });
});

describe('CORS', () => {
  it('answers a preflight from the site', async () => {
    const res = await worker.fetch(
      new Request('https://proxy.example/v1/score', {
        method: 'OPTIONS',
        headers: { origin: 'https://rejusamjohn.pages.dev' },
      }),
      env()
    );
    expect(res.status).toBe(204);
    expect(res.headers.get('access-control-allow-origin')).toBe('https://rejusamjohn.pages.dev');
    expect(res.headers.get('access-control-allow-methods')).toBe('GET, POST, OPTIONS');
    expect(res.headers.get('access-control-allow-headers')).toBe('Content-Type');
    expect(res.headers.get('access-control-max-age')).toBe('86400');
  });

  it('serves other origins without an allow-origin header', async () => {
    upstream(() => Response.json({ week: 8, results: [] }));
    const res = await worker.fetch(post(valid, { origin: 'https://evil.example' }), env());
    expect(res.status).toBe(200);
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
    expect(res.headers.get('vary')).toBe('Origin');
  });

  it('serves requests with no Origin (curl, Python)', async () => {
    upstream(() => Response.json({ week: 8, results: [] }));
    const res = await worker.fetch(post(valid), env());
    expect(res.status).toBe(200);
  });

  it('adds CORS headers to errors too', async () => {
    const res = await worker.fetch(
      post('', { origin: 'http://localhost:5500' }),
      env()
    );
    expect(res.status).toBe(400);
    expect(res.headers.get('access-control-allow-origin')).toBe('http://localhost:5500');
  });
});
