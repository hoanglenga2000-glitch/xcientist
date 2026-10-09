'use client';
import { ArrowLeft, ArrowRight, Check, Download, MessageSquare, Clock3, FileText } from 'lucide-react';
import type { PreviewTask } from '../model';

export const statusLabels = { draft: '尚未开始', needs_input: '需要你补充', completed: '样例文件已准备' };
export type DetailProps = {
  task: PreviewTask; tab: string; draft: string; busy: boolean; error: string;
  setDraft: (value: string) => void; send: () => void; reload: () => void;
};
export function TaskDetail({ task, tab, draft, busy, error, setDraft, send, reload }: DetailProps) {
  const link = (view: string) => `/?task=${encodeURIComponent(task.id)}&view=${view}`;
  return <>
    <a className="back-link" href="/"><ArrowLeft size={16} />我的任务</a>
    <header className="task-heading"><div><p className="eyebrow">当前任务 · 样例</p><h1>{task.title}</h1></div><span className={`status ${task.status}`}>{statusLabels[task.status]}</span></header>
    <nav className="task-tabs" aria-label="任务内容">
      {([['conversation', '对话', MessageSquare], ['progress', '进度', Clock3], ['results', '文件与结果', FileText]] as const).map(([view, label, Icon]) => <a key={view} href={link(view)} aria-current={tab === view ? 'page' : undefined}><Icon size={17} />{label}</a>)}
    </nav>
    {tab === 'conversation' && <section aria-label="任务对话" className="conversation">
      {!task.messages.length && <div className="empty-state"><MessageSquare size={28} /><h2>从你的需求开始</h2><p>草稿已创建，还没有执行任何工作。写下希望解决的问题，以及想收到什么结果。</p></div>}
      {task.messages.map((message, index) => <article key={index} className={`message ${message.role}`} aria-label={message.role === 'user' ? '你的消息' : '预览助手回复'}>
        <span className="message-author">{message.role === 'user' ? '你' : 'DeepEvo · 样例回复'}</span><p>{message.text}</p>
      </article>)}
      {task.result && <a className="result-callout" href={link('results')}><FileText size={20} /><span><strong>查看本任务的样例文件</strong><small>预览或下载，不代表研究已完成</small></span><ArrowRight size={18} /></a>}
      <form className="composer" onSubmit={event => { event.preventDefault(); send(); }}>
        <label htmlFor="request">输入需求</label>
        <textarea id="request" value={draft} maxLength={4000} onChange={event => setDraft(event.target.value)} placeholder="例如：整理工业缺陷检测的方法，给我一份阅读提纲。" disabled={busy} />
        <div className="composer-bottom"><span>仅演示交互 · 未调用模型<br />请勿输入密钥或敏感资料</span><button className="primary" disabled={busy || !draft.trim()} type="submit">{busy ? '正在保存…' : '查看样例回复'}<ArrowRight size={16} /></button></div>
        {error && <div role="alert" className="error"><p>{error}</p><button type="button" onClick={reload}>重新加载任务，保留输入</button></div>}
      </form>
    </section>}
    {tab === 'progress' && <section className="progress-pane"><h2>这项工作到哪一步了？</h2><p className="muted">只展示本任务已经记录的步骤，不用其他任务的进度补齐。</p>
      {!task.steps.length ? <div className="empty-state"><Clock3 size={28} /><h3>尚未开始</h3><p>草稿没有执行记录。回到对话，先描述你的需求。</p><a className="primary" href={link('conversation')}>去输入需求</a></div> : <ol className="timeline">{task.steps.map((step, index) => <li key={index}><span className={`step-dot ${step.state}`}>{step.state === 'done' ? <Check size={16} /> : index + 1}</span><div><h3>{step.title}</h3><p>{step.detail}</p>{step.state === 'waiting' && <a className="primary" href={link('conversation')}>去补充需求<ArrowRight size={15} /></a>}</div><span className="step-state">{step.state === 'done' ? '已记录' : '等你补充'}</span></li>)}</ol>}
      {task.result && <a className="primary" href={link('results')}>查看样例文件<ArrowRight size={16} /></a>}
      <details className="technical"><summary>技术详情</summary><p>来源：隔离样例会话。无真实模型、训练运行或外部服务调用。</p><code>{task.id}</code><p>记录版本：{task.version}</p></details>
    </section>}
    {tab === 'results' && <section className="results-pane"><h2>文件与结果</h2>{task.result ? <>
      <p className="muted">文件可下载。模型质量、科研结论和官方结果均未验证。</p>
      <div className="file-heading"><div><FileText size={24} /><span><strong>{task.result.name}</strong><small>本任务 · Markdown · 样例</small></span></div><a className="primary" href={`/api/preview/tasks/${encodeURIComponent(task.id)}/artifact`} download><Download size={16} />下载文件</a></div>
      <pre className="file-preview" aria-label="文件预览">{task.result.content}</pre>
      <details className="technical"><summary>来源与验证范围</summary><p>来源：预览程序生成或预先编写的样例文件。没有检索、模型生成或训练结果。所属样例任务：<code>{task.id}</code></p></details>
    </> : <div className="empty-state"><FileText size={28} /><h3>还没有结果文件</h3><p>完成相应步骤后，文件会出现在这里；当前没有可下载的成果。</p><a className="primary" href={link('conversation')}>回到对话</a></div>}</section>}
  </>;
}
