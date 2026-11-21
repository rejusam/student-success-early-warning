// Browsers may call the proxy only from the site and from Live Server on a laptop. Requests
// without an Origin header (curl, Python) are not browser requests, so CORS does not apply.
const ALLOWED = new Set(['https://rejusamjohn.pages.dev', 'http://localhost:5500']);

export function corsHeaders(origin: string | null): Record<string, string> {
  const headers: Record<string, string> = { Vary: 'Origin' };
  if (origin !== null && ALLOWED.has(origin)) headers['Access-Control-Allow-Origin'] = origin;
  return headers;
}
