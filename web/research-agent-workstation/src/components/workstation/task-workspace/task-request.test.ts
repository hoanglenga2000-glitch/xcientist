import test from 'node:test';
import assert from 'node:assert/strict';
// @ts-expect-error Native Node tests resolve the explicit .ts extension.
import { taskRequest } from './types.ts';

test('offline and timeout failures give an actionable message without replaying the request', async (context) => {
  const request = context.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(taskRequest('/api/assistant/tasks/owned/draft'), /网络连接中断.*输入.*保留/);
  assert.equal(request.mock.callCount(), 1);
  request.mock.mockImplementation(async () => { throw new DOMException('timed out', 'TimeoutError'); });
  await assert.rejects(taskRequest('/api/assistant/tasks/owned/draft'), /请求超时.*核对.*重试/);
  assert.equal(request.mock.callCount(), 2);
});
