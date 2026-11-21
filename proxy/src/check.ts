// Checks on a /v1/score body before it is sent to AWS. They stop oversized or malformed traffic
// early; the full rules (ranges, counts that must agree) stay in the API, which returns a 422
// with details when one is broken.
import limits from '../limits.json';

const WEEKS = [2, 4, 8, 12];
const MODULES = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'];
// The API's Student fields, in model order (src/early_warning/api.py).
const NUMBER_FIELDS = [
  'id_student',
  'clicks_total',
  'clicks_last_week',
  'active_days_total',
  'weeks_active',
  'clicks_before_start',
  'n_due',
  'n_submitted',
  'n_missed',
  'n_late',
  'mean_score',
  'num_of_prev_attempts',
  'studied_credits',
  'date_registration',
];
const NULLABLE = ['mean_score', 'date_registration'];
const FIELDS = [...NUMBER_FIELDS, 'code_module'];

export type ScoreBody = { week: number; students: Record<string, unknown>[] };
export type CheckResult =
  | { ok: true; body: ScoreBody }
  | { ok: false; status: 400 | 413 | 422; error: string };

const fail = (status: 400 | 413 | 422, error: string): CheckResult => ({ ok: false, status, error });
const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function checkScoreBody(raw: string): CheckResult {
  if (new TextEncoder().encode(raw).length > limits.max_body_bytes) {
    return fail(413, `The body is larger than ${limits.max_body_bytes} bytes.`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return fail(400, 'The body is not valid JSON.');
  }
  if (!isObject(parsed)) return fail(400, 'The body must be a JSON object with week and students.');

  for (const key of Object.keys(parsed)) {
    if (key !== 'week' && key !== 'students') return fail(422, `Unknown key: ${key}.`);
  }
  if (!WEEKS.includes(parsed.week as number)) return fail(422, 'week must be 2, 4, 8 or 12.');
  const students = parsed.students;
  if (!Array.isArray(students) || students.length < 1 || students.length > limits.max_students) {
    return fail(422, `students must be a list of 1 to ${limits.max_students} students.`);
  }

  for (const [i, student] of students.entries()) {
    if (!isObject(student)) return fail(422, `students[${i}] must be an object.`);
    for (const key of Object.keys(student)) {
      if (!FIELDS.includes(key)) return fail(422, `students[${i}] has an unknown key: ${key}.`);
    }
    for (const key of FIELDS) {
      if (!(key in student)) return fail(422, `students[${i}] is missing ${key}.`);
    }
    if (!MODULES.includes(student.code_module as string)) {
      return fail(422, `students[${i}].code_module must be one of ${MODULES.join(', ')}.`);
    }
    for (const key of NUMBER_FIELDS) {
      const value = student[key];
      if (value === null && NULLABLE.includes(key)) continue;
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        return fail(422, `students[${i}].${key} must be a number.`);
      }
    }
  }
  return { ok: true, body: parsed as ScoreBody };
}
