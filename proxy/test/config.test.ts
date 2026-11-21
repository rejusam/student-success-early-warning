import { describe, expect, it } from 'vitest';
import limits from '../limits.json';
import raw from '../wrangler.jsonc?raw';

const config = JSON.parse(
  raw
    .split('\n')
    .filter((line: string) => !line.trim().startsWith('//'))
    .join('\n')
);

describe('wrangler.jsonc', () => {
  it('uses the limits in limits.json', () => {
    const byName = Object.fromEntries(config.ratelimits.map((r: any) => [r.name, r.simple]));
    expect(byName.IP_LIMITER).toEqual({ limit: limits.ip_per_minute, period: 60 });
    expect(byName.UPSTREAM_LIMITER).toEqual({ limit: limits.upstream_per_minute, period: 60 });
  });

  it('points at the live scoring API', () => {
    expect(config.vars.AWS_API_URL).toBe('https://au3hc4ic73.execute-api.ap-southeast-2.amazonaws.com');
  });

  it('does not store invocation logs', () => {
    expect(config.observability.logs.invocation_logs).toBe(false);
  });
});
