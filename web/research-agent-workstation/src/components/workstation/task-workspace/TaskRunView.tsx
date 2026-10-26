'use client';
import { useState } from 'react';
import { FileText } from 'lucide-react';
import { restoreRunMessages } from '../screens/assistant-history';
import { actionName, blockedAssistantLine, blockedProgressCopy, formatFileSize, servedModelLabel, STREAM_PHASES, streamPhaseState, toolStepDetail, toolStepLabel, visibleMessageText } from './task-presentation';
import { taskStatus, type TaskArtifact, type TaskRun } from './types';
import { DownloadFile } from './DownloadFile';

function ResultFile({ file }: { file: TaskArtifact }) {
  const [preview, setPreview] = useState<string | null>(null);
  const [opened, setOpened] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const url = `/api/assistant/artifacts/${encodeURIComponent(file.id)}`;
  async function open() {
    setOpened(!opened); setError('');
    if (opened || preview !== null || ['pdf', 'image', 'svg', 'html'].includes(file.preview_kind) || file.bytes > 131072) return;
    setBusy(true);
    try {
      const response = await fetch(url, { cache: 'no-store', signal: AbortSignal.timeout(12000) });
      if (!response.ok) throw new Error(`预览失败（${response.status}），可重新打开重试。`);
      setPreview((await response.text()).slice(0, 131072));
    } catch (cause) { setError(cause instanceof Error ? cause.message : '预览失败，请重试。'); }
    finally { setBusy(false); }
  }
  return <article className="tw-file"><div className="tw-file-heading"><FileText size={22} aria-hidden="true" /><div><h2>{file.name}</h2><small>{formatFileSize(file.bytes)}</small></div>
    <button onClick={() => void open()} aria-expanded={opened}>{opened ? '关闭预览' : '预览'}</button><DownloadFile url={`${url}?download=1`} file={file} /></div>
    {opened && <div>{busy && <p role="status">正在读取文件…</p>}{error && <p role="alert" className="tw-error">{error}</p>}
      {['image', 'svg'].includes(file.preview_kind) ? <img className="tw-image" src={url} alt={file.name} /> : ['html', 'pdf'].includes(file.preview_kind) ? <iframe className="tw-document" title={file.name} src={url} sandbox="" /> : preview !== null ? <pre className="tw-file-preview">{preview}</pre> : !busy && !error ? <p className="tw-muted">此文件较大或为二进制格式，请下载后查看。</p> : null}</div>}
    <details className="tw-details"><summary>来源与校验</summary><dl><dt>生成操作</dt><dd>{actionName(file.source_tool_call)}</dd></dl></details>
  </article>;
}

function uniqueFiles(run: TaskRun) {
  const seen = new Set<string>();
  return [...(run.artifacts ?? []), ...(run.artifact_evidence ?? [])].filter(file => {
    if (!file.id || seen.has(file.id)) return false;
    seen.add(file.id);
    return true;
  });
}

function ChatProgress({ run, title }: { run: TaskRun; title?: string }) {
  const served = servedModelLabel(run.model_provider, run.model);
  const phase = streamPhaseState(run);
  const live = (run.plan?.steps ?? []).find(step => step.status === 'running');
  const liveText = live ? (toolStepDetail(live.detail) || toolStepLabel(live.label, live.status)) : '';
  const heading = phase.copy?.status || (phase.done ? '已交付' : (title || taskStatus(run.status)));
  return <article className="tw-progress-card" aria-label="任务进度">
    <header className="tw-progress-head"><span className={`tw-dot ${phase.stuck ? 'is-stuck' : phase.done ? 'is-done' : 'is-current'}`} /><strong>{heading}</strong><span className="tw-progress-model">本次实际模型 · {served || (run.model ? '平台模型' : '尚无记录')}</span></header>
    <ol className="tw-phases">{STREAM_PHASES.map((label, index) => {
      const cls = phase.done || index < phase.current ? 'is-done' : index === phase.current ? (phase.stuck ? 'is-stuck' : 'is-current') : '';
      return <li key={label} className={cls || undefined}><span className={cls ? `tw-dot ${cls}` : 'tw-dot'} />{label}</li>;
    })}</ol>
    {liveText && <p>{liveText}</p>}
    {phase.copy && <p>{phase.copy.action}</p>}
    <details className="tw-details"><summary>查看过程</summary>
      {(run.plan?.steps ?? []).length ? <ol className="tw-timeline">{run.plan!.steps!.map(step => {
        const detail = toolStepDetail(step.detail);
        return <li key={step.id}><div><h3>{toolStepLabel(step.label, step.status)}</h3>{detail && <p>{detail}</p>}</div></li>;
      })}</ol> : <p className="tw-muted">过程还没展开。</p>}
    </details>
  </article>;
}

export function TaskRunView({ view, run, title }: { view: string; run: TaskRun | null; title?: string }) {
  if (!run) return <div className="tw-empty"><h2>尚未开始</h2><p>保存需求不会开始执行。发送需求后，这里显示实际对话、进度和文件。</p></div>;
  const unavailable = run.status === 'blocked' && (run.error_class === 'model_unavailable' || Boolean(run.answer?.includes('Model execution is unavailable:')));
  const modelHelp = <div className="tw-error" role="alert"><h2>模型暂时不可用</h2><p>这次请求没有获得可用的模型回复。任务记录已保留；请检查服务地址、密钥和模型名称，再明确重试，不会自动改用其他模型。</p><a href={`/workspace?settings=models${run.user_task_id ? `&return_task=${encodeURIComponent(run.user_task_id)}` : ''}`}>检查模型配置</a><details className="tw-details"><summary>技术详情</summary><p>{run.error_class}</p><p>{run.error_message || run.answer}</p></details></div>;
  if (view === 'results') return <section>
    {(run.artifacts ?? []).map(file => <ResultFile key={file.id} file={file} />)}
    {!run.artifacts?.length && <div className="tw-empty"><h2>还没有结果文件</h2></div>}
    {Boolean(run.artifact_evidence?.length) && <details className="tw-details"><summary>中间文件与执行记录（{run.artifact_evidence!.length}）</summary>{run.artifact_evidence!.map(file => <ResultFile key={file.id} file={file} />)}</details>}
  </section>;
  const progressCopy = unavailable ? null : blockedProgressCopy(run);
  if (view === 'progress') return <section><h2>{progressCopy?.status || taskStatus(run.status)}</h2>
    {unavailable ? modelHelp : progressCopy && <p role="alert" className="tw-error">{progressCopy.action}</p>}
    <ol className="tw-timeline">{(run.plan?.steps ?? []).map(step => {
      const detail = toolStepDetail(step.detail);
      const statusLabel = ({ pending: '尚未执行', running: '正在处理', completed: '已完成', failed: '失败', skipped: '已跳过' } as Record<string, string>)[step.status] ?? toolStepLabel(step.status);
      return <li key={step.id}><div><h3>{toolStepLabel(step.label, step.status)}</h3>{detail && <p>{detail}</p>}</div><span>{statusLabel}</span></li>;
    })}</ol>
    {!run.plan?.steps?.length && <p className="tw-muted">当前运行尚未提供步骤记录，不推测正在执行的阶段。</p>}
    <details className="tw-details"><summary>运行信息</summary><dl><dt>实际模型</dt><dd>{run.model ? (servedModelLabel(run.model_provider, run.model) || '平台模型') : '尚无记录'}</dd></dl></details>
  </section>;
  const served = servedModelLabel(run.model_provider, run.model);
  const servedId = [run.model_provider, run.model].filter(Boolean).join('/');
  const generated = uniqueFiles(run);
  const visibleFiles = generated.slice(0, 2);
  const hiddenFiles = generated.slice(2);
  const phase = streamPhaseState(run);
  return <section aria-label="任务对话">{served && <p className="tw-served-model" data-served-model={servedId}>{served}</p>}
    {restoreRunMessages(run, '继续执行此任务').filter(message => message.content && !(unavailable && message.role === 'assistant' && message.content.includes('Model execution is unavailable:'))).map(message => {
      const content = visibleMessageText(message.content);
      return content ? <article className={`tw-message tw-${message.role}`} key={message.id}><small>{message.role === 'user' ? '你' : 'DeepEvo'}</small><p>{content}</p></article> : null;
    })}
    {phase.copy && <article className="tw-message tw-assistant"><small>DeepEvo</small><p>{blockedAssistantLine(phase.copy.status)}</p></article>}
    {!unavailable && <ChatProgress run={run} title={title} />}
    {visibleFiles.map(file => <ResultFile key={file.id} file={file} />)}
    {hiddenFiles.length > 0 && <details className="tw-details"><summary>还有 {hiddenFiles.length} 个文件</summary>{hiddenFiles.map(file => <ResultFile key={file.id} file={file} />)}</details>}
    {unavailable && modelHelp}
    {!run.answer && !run.streaming_text && !phase.copy && <p role="status" className="tw-muted">{taskStatus(run.status)}</p>}
  </section>;
}
