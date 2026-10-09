import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
// @ts-expect-error Native Node strip-types tests resolve explicit .ts extensions.
import { verifiedDownloadBytes } from './verified-download.ts';

test('download verifies actual bytes and refuses unauthorized or changed payloads', async () => {
  const data = new TextEncoder().encode('owned artifact');
  const expected = { bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') };
  assert.deepEqual(new Uint8Array(await verifiedDownloadBytes(new Response(data), expected)), data);
  await assert.rejects(verifiedDownloadBytes(new Response('{}', { status: 403 }), expected), /下载失败/);
  await assert.rejects(verifiedDownloadBytes(new Response(data), { ...expected, sha256: '0'.repeat(64) }), /校验未通过/);
  await assert.rejects(verifiedDownloadBytes(new Response('longer untrusted result'), expected), /大小/);
});
