import { describe, expect, it } from 'vitest';
import example from '../../docs/examples/score_request.json';
import { checkScoreBody } from '../src/check';

const student = example.students[0];
const body = (changes: Record<string, unknown> = {}, studentChanges: Record<string, unknown> = {}) =>
  JSON.stringify({ week: 8, students: [{ ...student, ...studentChanges }], ...changes });

function rejected(raw: string) {
  const result = checkScoreBody(raw);
  if (result.ok) throw new Error('expected a rejection');
  return result;
}

describe('checkScoreBody', () => {
  it('accepts the first example student', () => {
    const result = checkScoreBody(body());
    expect(result.ok).toBe(true);
  });

  it.each(['', 'null', '[]', '{', '"text"', '42'])('rejects %j as not a JSON object', (raw) => {
    expect(rejected(raw).status).toBe(400);
  });

  it('rejects a body over 8192 bytes', () => {
    expect(rejected('x'.repeat(8193)).status).toBe(413);
  });

  it('rejects a week the model does not have', () => {
    expect(rejected(body({ week: 3 })).status).toBe(422);
  });

  it('rejects 0 and 6 students', () => {
    expect(rejected(body({ students: [] })).status).toBe(422);
    expect(rejected(body({ students: Array(6).fill(student) })).status).toBe(422);
  });

  it('rejects a students value that is not a list of objects', () => {
    expect(rejected(body({ students: 'all' })).status).toBe(422);
    expect(rejected(body({ students: [null] })).status).toBe(422);
  });

  it('rejects an unknown key and names it', () => {
    const result = rejected(body({}, { gender: 'F' }));
    expect(result.status).toBe(422);
    expect(result.error).toContain('gender');
  });

  it('rejects an unknown top-level key', () => {
    expect(rejected(body({ extra: 1 })).status).toBe(422);
  });

  it('rejects a module code that is not in OULAD', () => {
    expect(rejected(body({}, { code_module: 'ZZZ' })).status).toBe(422);
  });

  it('rejects a string in a count field', () => {
    expect(rejected(body({}, { clicks_total: '529' })).status).toBe(422);
  });

  it('accepts null for mean_score and date_registration', () => {
    expect(checkScoreBody(body({}, { mean_score: null, date_registration: null })).ok).toBe(true);
  });

  it('rejects null in a field that needs a number', () => {
    expect(rejected(body({}, { n_due: null })).status).toBe(422);
  });

  it('rejects a missing required key and names it', () => {
    const { n_late, ...rest } = student;
    const result = rejected(JSON.stringify({ week: 8, students: [rest] }));
    expect(result.status).toBe(422);
    expect(result.error).toContain('n_late');
  });
});
