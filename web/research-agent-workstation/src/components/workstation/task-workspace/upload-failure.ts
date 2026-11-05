// t238 C7b COPY S1: the one action under a failed upload. A retry of quota_exceeded / disk_low_water always re-413s.
export type UploadFailureAction = '管理文件' | '稍后再试' | '刷新' | '重试';
export function uploadFailureAction(code: string | undefined): UploadFailureAction {
  if (code === 'quota_exceeded') return '管理文件';
  if (code === 'disk_low_water') return '稍后再试';
  if (code === 'upload_completing') return '刷新';
  return '重试';
}
