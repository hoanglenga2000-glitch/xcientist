'use client';
import { useEffect, useRef, useState, type FocusEvent, type KeyboardEvent } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import type { ModelList } from './ModelSettings';
import { productModelLabel } from './task-presentation';
import { extraPlatformModels, INTENSITY_LEVELS, PRODUCT_DEFAULT_MODEL, type Intensity } from './composer-options';

export type PickerChoice = { value: string; label: string; hint?: string; model?: string; section?: string };
export type PickerGroup = { label?: string; choices: PickerChoice[] };

const SECTIONS = ['Claude', 'GPT', 'Grok', '国模', '其他'] as const;
const CLAUDE_TIERS = ['opus', 'sonnet', 'haiku'];
const CLAUDE_SUFFIXES = ['', 'thinking', 'agentic'];

/** Family subheader inside 平台模型. */
export function modelSection(model: string) {
  const name = model.trim().toLowerCase();
  if (name.startsWith('claude')) return 'Claude';
  if (/^(?:gpt|o\d)/.test(name)) return 'GPT';
  if (name.startsWith('grok')) return 'Grok';
  if (/^(?:deepseek|kimi|qwen|glm|moonshot|doubao|minimax)/.test(name)) return '国模';
  return '其他';
}

/** Opus, Sonnet, Haiku; newest version first; a thinking or agentic sibling right after its base. */
function claudeKey(model: string) {
  const tokens = model.toLowerCase().replace(/-\d{8}$/, '').split(/[-.]/).slice(1);
  const tier = CLAUDE_TIERS.indexOf(tokens[0]);
  const version = tokens.slice(1).filter(token => /^\d+$/.test(token)).map(Number);
  const suffix = tokens.slice(1).filter(token => !/^\d+$/.test(token)).join('-');
  const rank = CLAUDE_SUFFIXES.indexOf(suffix);
  return { tier: tier < 0 ? CLAUDE_TIERS.length : tier, version, suffix: rank < 0 ? CLAUDE_SUFFIXES.length : rank };
}

function byClaude(left: string, right: string) {
  const a = claudeKey(left); const b = claudeKey(right);
  if (a.tier !== b.tier) return a.tier - b.tier;
  for (let i = 0; i < Math.max(a.version.length, b.version.length); i += 1) {
    const diff = (b.version[i] ?? 0) - (a.version[i] ?? 0);
    if (diff) return diff;
  }
  return a.suffix - b.suffix;
}

/** Default first, its family first; then Claude / GPT / Grok / 国模. Only Claude is re-ordered inside its family. */
function platformChoices(base: PickerChoice | null, extra: PickerChoice[]) {
  const order = [...new Set([base?.section, ...SECTIONS].filter((section): section is string => Boolean(section)))];
  return order.flatMap(section => {
    const rest = extra.filter(choice => choice.section === section);
    if (section === 'Claude') rest.sort((left, right) => byClaude(left.model || '', right.model || ''));
    return base?.section === section ? [base, ...rest] : rest;
  });
}

/** 平台模型 (default `platform`, then `platform:<id>`) and enabled 我的模型. Labels carry the exact version; ids stay in values. */
export function modelGroups(list: ModelList, platform?: { model?: string } | null): PickerGroup[] {
  const groups: PickerGroup[] = [];
  const key = (model?: string) => productModelLabel(model).trim().toLocaleLowerCase();
  // The default slot is always the product default. A run's model relabels it only when it is the same
  // product (e.g. a dated snapshot); any other model (a failed pick, a substitution) never replaces it (t237 C4.2).
  // 「· 默认」 therefore only ever names the product default; the last run's model stays on the run row and is
  // never added to this list (TaskSession passes run.model here under value === 'platform').
  const product = list.platform_model?.model || '';
  const served = platform?.model && key(platform.model) && key(platform.model) === key(product) ? platform.model : product;
  const base = list.platform_model ? { value: 'platform', label: `${productModelLabel(served) || '平台模型'} · 默认`, model: served, section: modelSection(served) } : null;
  // One model (or two ids with one product name) shows once; the default, then the resolver's first provider wins.
  // Only ids actually shown are seen, so no listed model is hidden by an unrelated run.
  const seenIds = new Set([product, served].filter(Boolean).map(model => String(model).toLocaleLowerCase()));
  const seenLabels = new Set(base ? [key(served)].filter(Boolean) : []);
  const extra = extraPlatformModels(list).filter(item => {
    const id = String(item.model || '').toLocaleLowerCase(); const label = key(item.model);
    if (seenIds.has(id) || (label && seenLabels.has(label))) return false;
    seenIds.add(id); if (label) seenLabels.add(label);
    return true;
  })
    .map(item => ({ value: 'platform:' + item.id, label: productModelLabel(item.model) || '平台模型', model: item.model, section: modelSection(item.model) }));
  if (base || extra.length) groups.push({ label: '平台模型', choices: platformChoices(base, extra) });
  const mine = list.profiles.filter(item => item.enabled).map(item => {
    const product = productModelLabel(item.model);
    return { value: item.id, label: item.name || product || '我的模型', hint: [item.name ? product : '', item.verified_at ? '' : '尚未测试'].filter(Boolean).join(' · ') };
  });
  if (mine.length) groups.push({ label: '我的模型', choices: mine });
  return groups;
}

const INTENSITY_HINTS: Record<Intensity, string> = { 低: '更快', 中: '默认', 高: '想得更久' };
const INTENSITY_GROUPS: PickerGroup[] = [{ choices: INTENSITY_LEVELS.map(level => ({ value: level, label: level, hint: INTENSITY_HINTS[level] })) }];

/** Composer popover listbox. Opens upward; the trigger keeps the id the form label points to. */
/** 「Claude Sonnet 5.5」 → name / version / rest: when the trigger is squeezed the brand ellipsises, the version never does. */
export function splitVersion(text: string): { name: string; version: string; rest: string } | null {
  const match = text.match(/^(\S.*?)(\s+\d+(?:\.\d+)*)(\s.*)?$/);
  return match ? { name: match[1], version: match[2], rest: match[3] ?? '' } : null;
}

export function Picker({ id, label, value, text, groups, disabled, autoOpen, closeSignal, footer, onChange }: {
  id: string; label: string; value: string; text: string; groups: PickerGroup[]; disabled?: boolean; autoOpen?: boolean;
  /** t238 C7c-A3 #1: any change closes an open picker (attachment error, upload start, training hint), so it never covers them. */
  closeSignal?: string;
  footer?: { href: string; label: string }; onChange: (value: string) => void;
}) {
  const flat = groups.flatMap(group => group.choices);
  const parts = splitVersion(text);
  // C6-brand S_b1: 「Claude Sonnet 5.5」 → narrow screens drop the vendor and read 「Sonnet 5.5」. t237 C6d: only 「Claude 」
  // before a tier name; any other first word (Grok Code, Kimi K2, a profile name) is the model's name and stays.
  const vendor = parts?.name.match(/^Claude\s+(?=(?:Opus|Sonnet|Haiku)\b)/i)?.[0] ?? '';
  // autoOpen opens once, as soon as there is something to choose; later renders never re-open.
  const autoOpened = useRef(false);
  const [open, setOpen] = useState(() => (autoOpened.current = Boolean(autoOpen && !disabled && flat.length)));
  const [active, setActive] = useState(() => Math.max(0, flat.findIndex(choice => choice.value === value)));
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const optionId = (index: number) => `${id}-option-${index}`;

  function show() {
    if (disabled || !flat.length) return;
    setActive(Math.max(0, flat.findIndex(choice => choice.value === value)));
    setOpen(true);
  }
  function close() {
    setOpen(false);
    trigger.current?.focus();
  }
  function choose(index: number) {
    const choice = flat[index];
    if (choice) onChange(choice.value);
    close();
  }
  useEffect(() => { if (open) list.current?.focus(); }, [open]);
  useEffect(() => { if (open) document.getElementById(optionId(active))?.scrollIntoView({ block: 'nearest' }); }, [open, active]);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);
  // t238 C7c-A3 #1: close (without moving focus) when the composer starts showing status / an error / a hint.
  const lastSignal = useRef(closeSignal);
  useEffect(() => {
    if (closeSignal === lastSignal.current) return;
    lastSignal.current = closeSignal;
    if (closeSignal) setOpen(false);
  }, [closeSignal]);
  // Outside click closes.
  useEffect(() => {
    if (!open) return undefined;
    function outside(event: PointerEvent) { if (!root.current?.contains(event.target as Node | null)) setOpen(false); }
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);
  useEffect(() => {
    if (!autoOpen || autoOpened.current || disabled || !flat.length) return;
    autoOpened.current = true;
    show();
  }, [autoOpen, disabled, flat.length]); // eslint-disable-line react-hooks/exhaustive-deps

  function onTriggerKey(event: KeyboardEvent) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); show(); }
  }
  function onListKey(event: KeyboardEvent) {
    const last = flat.length - 1;
    const next = ({ ArrowDown: Math.min(active + 1, last), ArrowUp: Math.max(active - 1, 0), Home: 0, End: last, PageDown: last, PageUp: 0 } as Record<string, number>)[event.key];
    if (next !== undefined) { event.preventDefault(); setActive(next); }
    else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(active); }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
    else if (event.key === 'Tab' && (event.shiftKey || !footer)) setOpen(false);
  }
  function onPopBlur(event: FocusEvent) {
    if (!root.current?.contains(event.relatedTarget as Node | null)) setOpen(false);
  }

  return <div className="tw-picker" ref={root}>
    <button type="button" ref={trigger} id={id} className="tw-picker-trigger" data-value={value} aria-haspopup="listbox" aria-expanded={open} aria-controls={`${id}-list`}
      aria-label={text.startsWith(label) ? text : `${label}：${text}`} title={text} disabled={disabled} onMouseDown={event => { if (open) event.preventDefault(); }} onClick={() => open ? close() : show()} onKeyDown={onTriggerKey}>
      {parts ? <span title={text} className="tw-picker-text"><span className="tw-picker-name">{vendor ? <><span className="tw-picker-vendor">{vendor}</span>{parts.name.slice(vendor.length)}</> : parts.name}</span><span className="tw-picker-version">{parts.version}</span>{parts.rest && <span className="tw-picker-rest">{parts.rest}</span>}</span>
        : <span title={text}>{text}</span>}<ChevronDown size={14} aria-hidden="true" />
    </button>
    <div className="tw-picker-pop" hidden={!open} onBlur={onPopBlur} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); close(); } }}>
      <div role="listbox" id={`${id}-list`} ref={list} tabIndex={-1} aria-label={label} aria-activedescendant={open ? optionId(active) : undefined} onKeyDown={onListKey}>
        {groups.map((group, groupIndex) => <div role="group" aria-label={group.label} key={group.label ?? groupIndex}>
          {group.label && <div className="tw-picker-group" aria-hidden="true">{group.label}</div>}
          {group.choices.map((choice, choiceIndex) => {
            const index = flat.indexOf(choice);
            const section = choice.section && choice.section !== group.choices[choiceIndex - 1]?.section
              ? <div className="tw-picker-section" key={'section-' + choice.section} aria-hidden="true">{choice.section}</div> : null;
            const title = [choice.label, choice.hint].filter(Boolean).join(' · ');
            return [section, <div role="option" id={optionId(index)} key={choice.value} data-value={choice.value} aria-selected={choice.value === value} title={title} className={open && index === active ? 'is-active' : undefined}
              onPointerMove={() => setActive(index)} onClick={() => choose(index)}>
              <span className="tw-picker-label" title={title}>{choice.label}</span>{choice.hint && <small>{choice.hint}</small>}
              {choice.value === value && <Check size={14} aria-hidden="true" />}
            </div>];
          })}
        </div>)}
      </div>
      {footer && <div className="tw-picker-footer"><a href={footer.href}>{footer.label}</a></div>}
    </div>
  </div>;
}

export function ModelPicker({ id, list, value, platform, disabled, autoOpen, closeSignal, returnTask = '', onChange }: {
  id: string; list: ModelList; value: string; platform?: { model?: string } | null; disabled?: boolean; autoOpen?: boolean; closeSignal?: string; returnTask?: string; onChange: (value: string) => void;
}) {
  const groups = modelGroups(list, platform);
  const current = groups.flatMap(group => group.choices).find(choice => choice.value === value);
  // Before /api/assistant/model-profiles loads (no platform_model key yet) name the product default, not 「选择模型」.
  const loading = !('platform_model' in list) && !list.profiles.length;
  const text = (current?.model && productModelLabel(current.model)) || current?.label || (loading ? productModelLabel(PRODUCT_DEFAULT_MODEL) : '选择模型');
  const footer = { href: `/workspace?settings=models${returnTask ? `&return_task=${encodeURIComponent(returnTask)}` : ''}`, label: '配置我的模型' };
  return <Picker id={id} label="本次使用的模型" value={value} text={text} groups={groups} disabled={disabled} autoOpen={autoOpen} closeSignal={closeSignal} footer={footer} onChange={onChange} />;
}

export function IntensityPicker({ id, value, disabled, onChange }: { id: string; value: Intensity; disabled?: boolean; onChange: (value: Intensity) => void }) {
  return <Picker id={id} label="强度" value={value} text={`强度 · ${value}`} groups={INTENSITY_GROUPS} disabled={disabled} onChange={next => onChange(next as Intensity)} />;
}
