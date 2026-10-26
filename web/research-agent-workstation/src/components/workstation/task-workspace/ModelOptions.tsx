import type { ModelList } from './ModelSettings';
import { productModelLabel } from './task-presentation';
import { extraPlatformModels } from './composer-options';

/** 平台模型 only when the API serves one; 我的模型 lists enabled profiles. */
export function ModelOptions({ list, platformLabel, value }: { list: ModelList; platformLabel: string; value: string }) {
  const mine = list.profiles.filter(item => item.enabled);
  const extra = extraPlatformModels(list);
  return <>
    {!value && <option value="" disabled hidden>请选择模型</option>}
    {(list.platform_model || extra.length > 0) && <optgroup label="平台模型">
      {list.platform_model && <option value="platform">{platformLabel}</option>}
      {extra.map(item => <option key={item.id} value={'platform:' + item.id}>{productModelLabel(item.model) || item.label || '平台模型'}</option>)}
    </optgroup>}
    {mine.length > 0 && <optgroup label="我的模型">{mine.map(item => <option key={item.id} value={item.id}>{[item.name, productModelLabel(item.model)].filter(Boolean).join(' · ') || '我的模型'}{item.verified_at ? '' : ' · 尚未测试'}</option>)}</optgroup>}
  </>;
}
