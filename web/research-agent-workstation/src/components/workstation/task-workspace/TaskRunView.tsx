'use client';
import { useState } from 'react';
import { FileText } from 'lucide-react';
import { restoreRunMessages } from '../screens/assistant-history';
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
    <details className="tw-details"><summary>来源与校验</summary><p>文件属于本次运行。文件校验不代表科研结论或模型质量通过。</p><dl><dt>运行</dt><dd>{file.run_id}</dd><dt>SHA-256</dt><dd>{file.sha256}</dd><dt>最近文件校验</dt><dd>{file.verified_at || '未提供验证时间'}</dd><dt>生成操作</dt><dd>{file.source_tool_call || '未提供'}</dd></dl></details>
  </article>;
}

export function TaskRunView({ view, run }: { view: string; run: TaskRun | null }) {
  if (!run) return <div className="tw-empty"><h2>尚未开始</h2><p>保存需求不会开始执行。发送需求后，这里显示实际对话、进度和文件。</p></div>;
  const unavailable = run.status === 'blocked' && (run.error_class === 'model_unavailable' || Boolean(run.answer?.includes('Model execution is unavailable:')));
  const modelHelp = <div className="tw-error" role="alert"><h2>模型暂时不可用</h2><p>这次请求没有获得可用的模型回复。任务记录已保留；请检查服务地址、密钥和模型名称，再明确重试，不会自动改用其他模型。</p><a href={`/workspace?settings=models${run.user_task_id ? `&return_task=${encodeURIComponent(run.user_task_id)}` : ''}`}>检查模型配置</a><details className="tw-details"><summary>技术详情</summary><p>{run.error_class}</p><p>{run.error_message || run.answer}</p></details></div>;
  if (view === 'results') return <section><p className="tw-muted">以下文件来自所选运行，也可能包含诊断记录。文件存在不代表任务目标完成或科研结论已确认。</p>
    {(run.artifacts ?? []).map(file => <ResultFile key={file.id} file={file} />)}
    {!run.artifacts?.length && <div className="tw-empty"><h2>还没有结果文件</h2><p>可以回到对话查看回复或说明希望输出的文件。系统不会用示例文件填充这里。</p></div>}
    {Boolean(run.artifact_evidence?.length) && <details className="tw-details"><summary>中间文件与执行记录（{run.artifact_evidence!.length}）</summary>{run.artifact_evidence!.map(file => <ResultFile key={file.id} file={file} />)}</details>}
  </section>;
  if (view === 'progress') return <section><h2>{taskStatus(run.status)}</h2>
    {unavailable ? modelHelp : run.error_class && <div role="alert" className="tw-error"><p>{run.error_message || '本次执行遇到问题。请查看详情，处理后再明确选择继续。'}</p><details><summary>技术错误</summary>{run.error_class}</details></div>}
    <ol className="tw-timeline">{(run.plan?.steps ?? []).map(step => <li key={step.id}><div><h3>{step.label}</h3>{step.detail && <p>{step.detail}</p>}</div><span>{({ pending: '尚未执行', running: '正在处理', completed: '已完成', failed: '失败', skipped: '已跳过' } as Record<string, string>)[step.status] ?? step.status}</span></li>)}</ol>
    {!run.plan?.steps?.length && <p className="tw-muted">当前运行尚未提供步骤记录，不推测正在执行的阶段。</p>}
    <details className="tw-details"><summary>运行信息</summary><dl><dt>运行编号</dt><dd>{run.id}</dd><dt>实际模型</dt><dd>{run.model ? `${run.model_provider ?? ''} / ${run.model}` : '尚无实际模型执行记录'}</dd></dl></details>
  </section>;
  return <section aria-label="任务对话">{restoreRunMessages(run, '继续执行此任务').filter(message => message.content && !(unavailable && message.role === 'assistant' && message.content.includes('Model execution is unavailable:'))).map(message => <article className={`tw-message tw-${message.role}`} key={message.id}><small>{message.role === 'user' ? '你' : 'DeepEvo'}</small><p>{message.content}</p></article>)}{unavailable && modelHelp}
    {!run.answer && !run.streaming_text && <p role="status" className="tw-muted">{taskStatus(run.status)}。进度页可查看当前步骤与原因。</p>}
  </section>;
}
