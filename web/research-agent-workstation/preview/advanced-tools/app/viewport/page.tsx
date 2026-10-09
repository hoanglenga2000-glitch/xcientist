import { notFound } from 'next/navigation';
export const dynamic = 'force-dynamic';
export default async function Viewport({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
  if (process.env.EVOMIND_USABILITY_PREVIEW !== '1') notFound();
  const params = await searchParams;
  const sizes: Record<string, [number, number]> = { mobile: [390, 844], desktop: [1366, 768], wide: [1920, 1080] };
  const [width, height] = sizes[params.size ?? 'mobile'] ?? sizes.mobile;
  const target = params.task ? `/?task=${encodeURIComponent(params.task)}&view=${params.view === 'results' ? 'results' : params.view === 'progress' ? 'progress' : 'conversation'}` : '/';
  return <div style={{ padding: 16 }}><p>响应式验收 · {width} × {height} CSS 视口（不是手机设备模拟）</p><nav style={{ display: 'flex', gap: 20, marginBottom: 16 }}><a href="/viewport?size=mobile">手机宽度</a><a href="/viewport?size=desktop">桌面</a><a href="/viewport?size=wide">宽屏</a><a href="/">返回正常预览</a></nav><iframe title="响应式任务预览" src={target} width={width} height={height} style={{ display: 'block', border: '1px solid #305650', maxWidth: 'none' }} /></div>;
}
