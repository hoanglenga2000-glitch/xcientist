import { ArrowUpRight, ListTodo } from 'lucide-react';
import { taskStatus, type UserTask } from './types';

export function TaskList({ tasks, query }: { tasks: UserTask[]; query: string }) {
  const visible = tasks.filter(task => task.title.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  return <div className="tw-task-list">
    {visible.map(task => <a className="tw-task-row" key={task.id} href={`/workspace?task=${encodeURIComponent(task.id)}`}>
      <ListTodo size={22} aria-hidden="true" /><div className="tw-task-summary"><h2>{task.title}</h2>
        <p>{task.status === 'draft' ? '需求已保存，打开后继续' : '打开对话、进度和文件与结果'}</p></div>
      <div className="tw-task-meta"><span>{taskStatus(task.status)}</span><time dateTime={task.updated_at}>{new Date(task.updated_at).toLocaleDateString('zh-CN')}</time></div>
      <ArrowUpRight size={18} aria-hidden="true" /></a>)}
    {!visible.length && <div className="tw-empty"><h2>{query ? '没有匹配的任务' : '还没有任务'}</h2>
      <p>{query ? '换一个关键词试试。' : '先描述你想完成什么，不需要填写任务编号。'}</p><a className="tw-primary" href="/workspace?new=1">新任务</a></div>}
  </div>;
}
