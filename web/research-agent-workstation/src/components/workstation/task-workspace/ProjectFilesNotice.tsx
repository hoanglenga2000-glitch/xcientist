'use client';
import type { TaskRun } from './types';

/** t237 C3.10 (spec 05 M4): project files the run could not read are named under the answer, never guessed around. */
export type ProjectFileUnreadable = { name: string; error_class: string };
export type ProjectFilesNoticeValue = { kind: 'partial' | 'total'; files: ProjectFileUnreadable[] };

export function projectFilesNotice(run: Pick<TaskRun, 'project_files_unreadable' | 'project_files_total' | 'project_files_failed'>): ProjectFilesNoticeValue | null {
  const files = run.project_files_unreadable ?? [];
  if (run.project_files_failed) return { kind: 'total', files };
  if (!files.length) return null;
  return { kind: run.project_files_total !== undefined && files.length >= run.project_files_total ? 'total' : 'partial', files };
}

export function ProjectFilesNotice({ notice, projectId, onRetry }: { notice: ProjectFilesNoticeValue; projectId?: string | null; onRetry?: () => void }) {
  const details = notice.files.length > 0 && <details className="tw-details"><summary>技术详情</summary>
    {notice.files.map((file, index) => <p key={index}>{file.name} · {file.error_class}</p>)}</details>;
  if (notice.kind === 'total') return <div role="alert" className="tw-error tw-project-files-notice"><p>项目文件暂时读不了，这次回答没有用到。</p>
    <button type="button" onClick={() => onRetry ? onRetry() : window.location.reload()}>重试</button>{details}</div>;
  return <div className="tw-project-files-notice"><p className="tw-muted">有 {notice.files.length} 个项目文件没能读取，这次回答没有用到。</p>
    {projectId && <a href={`/workspace?project=${encodeURIComponent(projectId)}`}>查看项目文件</a>}{details}</div>;
}
