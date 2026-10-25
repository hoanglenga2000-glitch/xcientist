'use client';
import { useState } from 'react';
import { FileText } from 'lucide-react';
import { restoreRunMessages } from '../screens/assistant-history';
import { actionName, servedModelLabel, toolStepDetail, toolStepLabel, visibleMessageText } from './task-presentation';
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
  return <article className="tw-file"><div className="tw-file-heading"><FileText size={22} aria-hidden="true" /><div><h2>{file.name}</h2><small>{file.bytes.toLocaleString()} 字节</small></div>
    <button onClick={() => void open()} aria-expanded={opened}>{opened ? '关闭预览' : '预览'}</button><DownloadFile url={`${url}?download=1`} file={file} /></div>
    {opened && <div>{busy && <p role="status">正在读取文件…</p>}{error && <p role="alert" className="tw-error">{error}</p>}
      {['image', 'svg'].includes(file.preview_kind) ? <img className="tw-image" src={url} alt={file.name} /> : ['html', 'pdf'].includes(file.preview_kind) ? <iframe className="tw-document" title={file.name} src={url} sandbox="" /> : preview !== null ? <pre className="tw-file-preview">{preview}</pre> : !busy && !error ? <p className="tw-muted">此文件较大或为二进制格式，请下载后查看。</p> : null}</div>}
    <details className="tw-details"><summary>来源与校验</summary><dl><dt>生成操作</dt><dd>{actionName(file.source_tool_call)}</dd></dl></details>
  </article>;
}

export function TaskRunView({ view, run }: { view: string; run: TaskRun | null }) {
  if (!run) return <div className="tw-empty"><h2>尚未开始</h2><p>保存需求不会开始执行。发送需求后，这里显示实际对话、进度和文件。</p></div>;
  const unavailable = run.status === 'blocked' && (run.error_class === 'model_unavailable' || Boolean(run.answer?.includes('Model execution is unavailable:')));
  const modelHelp = <div className="tw-error" role="alert"><h2>模型暂时不可用</h2><p>这次请求没有获得可用的模型回复。任务记录已保留；请检查服务地址、密钥和模型名称，再明确重试，不会自动改用其他模型。</p><a href={`/workspace?settings=models${run.user_task_id ? `&return_task=${encodeURIComponent(run.user_task_id)}` : ''}`}>检查模型配置</a><details className="tw-details"><summary>技术详情</summary><p>{run.error_class}</p><p>{run.error_message || run.answer}</p></details></div>;
  if (view === 'results') return <section>
    {(run.artifacts ?? []).map(file => <ResultFile key={file.id} file={file} />)}
    {!run.artifacts?.length && <div className="tw-empty"><h2>还没有结果文件</h2></div>}
    {Boolean(run.artifact_evidence?.length) && <details className="tw-details"><summary>中间文件与执行记录（{run.artifact_evidence!.length}）</summary>{run.artifact_evidence!.map(file => <ResultFile key={file.id} file={file} />)}</details>}
  </section>;
  if (view === 'progress') return <section><h2>{taskStatus(run.status)}</h2>
    {unavailable ? modelHelp : run.error_class && <div role="alert" className="tw-error"><p>{run.error_message || '本次执行遇到问题。请查看详情，处理后再明确选择继续。'}</p><details><summary>技术错误</summary>{run.error_class}</details></div>}
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
  const generated = [...(run.artifacts ?? []), ...(run.artifact_evidence ?? [])];
  return <section aria-label="任务对话">{served && <p className="tw-served-model" data-served-model={servedId}>{served}</p>}
    {generated.map(file => <ResultFile key={file.id} file={file} />)}
    {restoreRunMessages(run, '继续执行此任务').filter(message => message.content && !(unavailable && message.role === 'assistant' && message.content.includes('Model execution is unavailable:'))).map(message => {
      const content = visibleMessageText(message.content);
      return content ? <article className={`tw-message tw-${message.role}`} key={message.id}><small>{message.role === 'user' ? '你' : 'DeepEvo'}</small><p>{content}</p></article> : null;
    })}{unavailable && modelHelp}
    {!run.answer && !run.streaming_text && <p role="status" className="tw-muted">{taskStatus(run.status)}。进度页可查看当前步骤与原因。</p>}
  </section>;
}
