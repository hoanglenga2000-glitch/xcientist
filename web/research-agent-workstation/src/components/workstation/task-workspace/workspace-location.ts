// Old identifiers stay distinct until an authenticated lookup verifies a link.
export function workspaceLocation(params: URLSearchParams) {
  const target = new URLSearchParams();
  const page = params.get('page') || 'tasks';
  const sections: Record<string, string> = { data: 'data', literature: 'literature', report: 'results', projects: 'projects', experiments: 'results' };
  if (['settings', 'design', 'gpu'].includes(page)) target.set('settings', page === 'gpu' ? 'resources' : 'models');
  if (sections[page]) target.set('section', sections[page]);
  const task = params.get('task') || '';
  if (/^utask_[a-f0-9]{32}$/.test(task)) target.set('task', task);
  else if (task) target.set('legacy_task', task.slice(0, 128));
  const run = params.get('assistant_run') || params.get('report_run') || params.get('run') || '';
  if (run) target.set(target.has('task') ? 'assistant_run' : 'legacy_run', run.slice(0, 128));
  const returnTask = params.get('return_task') || (target.has('task') ? task : '');
  if (target.has('settings') && /^utask_[a-f0-9]{32}$/.test(returnTask)) target.set('return_task', returnTask);
  const views: Record<string, string> = { workflow: 'progress', runtime: 'progress', gates: 'progress', code: 'results', evidence: 'results', report: 'results', evolution: 'more' };
  if (views[page]) target.set('view', views[page]);
  if (['code', 'evolution', 'gates', 'experiments'].includes(page)) target.set('intent', page);
  if (!task && ['control', 'assistant'].includes(page) && !run) target.set('new', '1');
  return '/workspace' + (target.size ? '?' + target.toString() : '');
}
