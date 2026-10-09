import type { ReactNode } from 'react';
import './preview.css';
export const metadata = { title: 'DeepEvo · 任务体验预览' };
export default function Layout({ children }: { children: ReactNode }) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
