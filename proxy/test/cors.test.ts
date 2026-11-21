import { describe, expect, it } from 'vitest';
import { corsHeaders } from '../src/cors';

describe('corsHeaders', () => {
  it.each(['https://rejusamjohn.pages.dev', 'http://localhost:5500'])('allows %s', (origin) => {
    expect(corsHeaders(origin)).toEqual({ 'Access-Control-Allow-Origin': origin, Vary: 'Origin' });
  });

  it.each([null, 'https://evil.example', 'https://rejusamjohn.pages.dev.evil.example'])(
    'gives no allow header for %s',
    (origin) => {
      expect(corsHeaders(origin)).toEqual({ Vary: 'Origin' });
    }
  );
});
