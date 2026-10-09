import { notFound } from 'next/navigation';
import { PreviewWorkspace } from '../components/PreviewWorkspace';
export const dynamic = 'force-dynamic';
export default function Page() {
  if (process.env.EVOMIND_USABILITY_PREVIEW !== '1') notFound();
  return <PreviewWorkspace />;
}
