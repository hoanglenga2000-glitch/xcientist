export type PreviewTask = {
  id: string;
  title: string;
  sample: true;
  version: number;
  status: 'draft' | 'needs_input' | 'completed';
  updated: string;
  messages: { role: 'user' | 'assistant'; text: string }[];
  steps: { title: string; state: 'done' | 'waiting'; detail: string }[];
  result: null | { name: string; content: string };
};

// Explicit fixtures, never a fallback for a missing production task or provider.
export function sampleTasks(): PreviewTask[] {
  return [
    {
      id: 'sample-review', title: '了解小样本图像分类的方法', sample: true,
      version: 1, status: 'needs_input', updated: '2026-10-08T09:30:00+08:00',
      messages: [
        { role: 'user', text: '帮我梳理小样本图像分类的研究方法，先给一份阅读提纲。' },
        { role: 'assistant', text: '这项工作可以先分成方法分类、阅读顺序和比较维度。请补充你关注的应用场景，例如医学图像或工业缺陷检测。\n\n这是预先编写的交互样例，还没有搜索论文或调用模型。' },
      ],
      steps: [
        { title: '明确研究需求', state: 'done', detail: '已保存样例中的研究目标。' },
        { title: '补充应用场景', state: 'waiting', detail: '需要你补充：研究用于什么场景。可以回到对话继续输入。' },
      ], result: null,
    },
    {
      id: 'sample-report', title: '整理实验结果', sample: true,
      version: 1, status: 'completed', updated: '2026-10-08T10:00:00+08:00',
      messages: [
        { role: 'user', text: '给我一份实验记录模板，方便之后填写实际测量结果。' },
        { role: 'assistant', text: '样例文件已准备好。打开“文件与结果”即可预览和下载。文件没有实验分数，也不代表实验已执行。' },
      ],
      steps: [
        { title: '明确交付内容', state: 'done', detail: '需要一份实验记录模板。' },
        { title: '准备样例文件', state: 'done', detail: '预先编写的 Markdown 文件，可预览和下载；未运行实验。' },
      ],
      result: { name: 'experiment-notes-sample.md', content: '# 实验记录模板（样例）\n\n所属任务：整理实验结果\n样例标识：sample-report\n\n本文件为隔离交互预览预先编写，未调用模型、未执行实验。\n\n## 待填写\n- 数据集与划分：未填写\n- 方法与配置：未填写\n- 指标与方向：未填写\n- 实测结果：未填写\n\n文件可下载，不代表模型质量或科研结论已验证。\n' },
    },
  ];
}
