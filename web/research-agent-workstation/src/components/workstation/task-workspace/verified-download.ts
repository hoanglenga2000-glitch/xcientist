export async function verifiedDownloadBytes(response: Response, expected: { bytes: number; sha256: string }): Promise<ArrayBuffer> {
  if (!response.ok) throw new Error(response.status === 401 ? '登录已过期，请重新登录后下载。' : '下载失败，当前任务保留，请重试。');
  if (expected.bytes > 128 * 1024 * 1024) throw new Error('此候选的页内校验下载上限为 128 MiB，请联系管理员安全导出较大文件。');
  const reader = response.body?.getReader();
  if (!reader) throw new Error('下载没有返回文件内容，请重试。');
  const chunks: Uint8Array[] = []; let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      length += value.length;
      if (length > expected.bytes) { await reader.cancel(); throw new Error('文件大小与记录不一致，已停止下载。'); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const data = new Uint8Array(length); let offset = 0;
  for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.length; }
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), byte => byte.toString(16).padStart(2, '0')).join('');
  if (length !== expected.bytes || hash !== expected.sha256) throw new Error('文件校验未通过，未保存错误文件。请重新读取任务记录后重试。');
  return data.buffer;
}
