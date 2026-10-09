import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { hoursToSeconds, validateBudget } from './execution-budget.ts';

test('hours convert to integer seconds without empty NaN or signed values', () => {
  assert.equal(hoursToSeconds('12'), 43200);
  assert.equal(hoursToSeconds('0.0333'), 120);
  for (const value of ['', ' ', '-1', 'NaN', 'Infinity', '1e8', '0.12345']) assert.throws(() => hoursToSeconds(value));
});
test('suballocation cannot exceed total or erase committed reservations', () => {
  const current = { gpu_limit_seconds: 43200, engineering_limit_seconds: 28800, committed_seconds: 39000, engineering_committed_seconds: 39000 };
  assert.equal(validateBudget(50000, 40000, current, 31536000), '');
  assert.equal(validateBudget(50000, 38000, current, 31536000), 'budget_below_committed');
  assert.equal(validateBudget(40000, 50000, current, 31536000), 'budget_limits_invalid');
});
test('budget UI uses managed endpoint and keeps explicit authorization and revision', () => {
  const ui = readFileSync(new URL('./ExecutionBudgetPanel.tsx', import.meta.url), 'utf8');
  const route = readFileSync(new URL('../../../app/api/assistant/execution-budget/route.ts', import.meta.url), 'utf8');
  assert.match(ui, /expected_revision: data.budget.revision/);
  assert.match(ui, /request_id: requestRef.current.id/);
  assert.match(ui, /!confirmed/);
  assert.match(ui, /setConfirmed\(false\)/);
  assert.match(route, /proxyAssistantRuntime/);
  assert.doesNotMatch(ui, /localStorage|gpu_budget.sqlite3|study_id:/);
});
