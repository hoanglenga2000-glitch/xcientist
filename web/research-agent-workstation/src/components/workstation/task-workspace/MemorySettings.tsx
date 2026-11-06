'use client';
import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Check, EyeOff, MoreHorizontal, Pencil, Pin, Trash2 } from 'lucide-react';
import { MenuItem, type MenuAnchor } from './ConversationMenu';
import { PopMenu } from './ProjectMenu';
import { showToast } from './Toast';
import { useReturnFocus } from './focus-return';
import {
  EXPERIENCE_COPY, MEMORY_COPY, MEMORY_HIGHLIGHT_MS, MEMORY_KINDS, MEMORY_KIND_LABEL, MEMORY_TABS, clampMemoryText, clearMemories, deleteExperience, deleteMemory, filterMemories,
  getMemorySettings, getProjectMemorySettings, listMemories, memoryChipLabel, memoryChipTitle, memoryDate, memoryHref, memoryKindLabel, memorySource,
  listExperiences, memorySourceHref, memoryUsedLabel, memoryUsedText, mergeEditedMemory, patchMemory, resolveMemoryEnabled, restoreMemory, setExperienceContribution,
  setMemoryEnabled, setMemoryPaused, setProjectMemoryOnly, type Experience, type Memory, type MemoryKind, type MemorySaved, type MemoryTab,
} from './memory';

const MENU_ID = 'tw-memory-menu';
type Editing = { id: string; value: string; truncated: boolean; kind: MemoryKind };

/** 已记住 under an assistant answer; opens 我的 → 记忆 and flashes the first saved row. */
export function MemoryChip({ saved }: { saved: MemorySaved[] }) {
  if (!saved.length) return null;
  return <a className="tw-chip tw-memory-chip" href={memoryHref(saved[0].id)} title={memoryChipTitle(saved)} aria-label={`已记住：${saved.map(item => item.text).join('；')}。管理记忆`}>
    <Check size={12} aria-hidden="true" />{memoryChipLabel(saved)}</a>;
}

/** C6c H7-5: 「用到的记忆 · N」 folded under the answer; short lines, never ids. Nothing injected ⇒ nothing rendered. */
export function MemoryUsed({ used }: { used: MemorySaved[] }) {
  if (!used.length) return null;
  return <details className="tw-memory-used" data-testid="tw-memory-used"><summary>{memoryUsedLabel(used.length)}</summary>
    <ul>{used.map(item => <li key={item.id}>{memoryUsedText(item.text)}</li>)}</ul></details>;
}

/** C6c CP7: 无痕 = this run reads and writes no memory. */
export const MEMORY_MODE_INCOGNITO = 'incognito';

/** H7-2: a secondary outline control, never a solid accent button; one click on, one click off (H7-4). */
export function IncognitoToggle({ on, disabled = false, onChange }: { on: boolean; disabled?: boolean; onChange: (on: boolean) => void }) {
  return <button type="button" className="tw-incognito" aria-pressed={on} data-testid="tw-incognito" disabled={disabled} onClick={() => onChange(!on)}>
    <EyeOff size={14} aria-hidden="true" />{MEMORY_COPY.incognito}</button>;
}

/** H7-3: one muted status line while 无痕 is on. 无痕 is the stricter state, so it is the only line even when memory is paused (S7-1). */
export function IncognitoStatus({ on }: { on: boolean }) {
  return on ? <p className="tw-incognito-status" role="status">{MEMORY_COPY.incognitoStatus}</p> : null;
}

export function MemoryRowMenu({ onEdit, onDelete }: { onEdit: () => void; onDelete: () => void }) {
  return <><MenuItem icon={<Pencil size={16} aria-hidden="true" />} label={MEMORY_COPY.edit} onSelect={onEdit} />
    <MenuItem icon={<Trash2 size={16} aria-hidden="true" />} label={MEMORY_COPY.remove} danger onSelect={onDelete} /></>;
}

/** Over 2000 is truncated on input and the tip stays; blank is refused (保存 disabled). C6c: 分类 can be changed here. */
export function MemoryEditor({ value, truncated, kind = 'other', busy = false, onChange, onKind, onSave, onCancel }: {
  value: string; truncated: boolean; kind?: MemoryKind; busy?: boolean; onChange: (value: string, truncated: boolean) => void; onKind?: (kind: MemoryKind) => void;
  onSave: () => void; onCancel: () => void;
}) {
  return <div className="tw-memory-edit">
    <textarea aria-label={MEMORY_COPY.editLabel} value={value} autoFocus
      onChange={event => { const [text, cut] = clampMemoryText(event.target.value); onChange(text, cut); }} />
    {truncated && <p className="tw-muted" role="status">{MEMORY_COPY.limit}</p>}
    {onKind && <select aria-label={MEMORY_COPY.kind} className="tw-memory-kind-select" value={kind} onChange={event => onKind(event.target.value as MemoryKind)}>
      {MEMORY_KINDS.map(item => <option key={item} value={item}>{MEMORY_KIND_LABEL[item]}</option>)}</select>}
    <div className="tw-actions"><button type="button" className="tw-primary" disabled={busy || !value.trim()} onClick={onSave}>{MEMORY_COPY.save}</button>
      <button type="button" onClick={onCancel}>{MEMORY_COPY.cancel}</button></div>
  </div>;
}

/** H6-5: 取消 is focused first, Tab stays inside, Esc cancels; 清空 is the danger button. No undo exists for this action. */
export function ClearAllDialog({ onCancel, onConfirm }: { count?: number; onCancel: () => void; onConfirm: () => void }) {
  useReturnFocus();
  const cancel = useRef<HTMLButtonElement>(null);
  const confirm = useRef<HTMLButtonElement>(null);
  useEffect(() => { cancel.current?.focus(); }, []);
  function keys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onCancel(); }
    else if (event.key === 'Tab') { event.preventDefault(); (document.activeElement === cancel.current ? confirm.current : cancel.current)?.focus(); }
  }
  return <div className="tw-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); onCancel(); } }}>
    <div className="tw-dialog" role="alertdialog" aria-modal="true" aria-labelledby="tw-clear-memory-title" aria-describedby="tw-clear-memory-line" onKeyDown={keys}>
      <h2 id="tw-clear-memory-title">{MEMORY_COPY.clearTitle}</h2><p id="tw-clear-memory-line">{MEMORY_COPY.clearLine}</p>
      <div className="tw-dialog-actions"><button ref={cancel} type="button" className="tw-button" onClick={onCancel}>{MEMORY_COPY.cancel}</button>
        <button ref={confirm} type="button" className="tw-button tw-danger-solid" onClick={onConfirm}>{MEMORY_COPY.clear}</button></div>
    </div></div>;
}

function Switch({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <div className="tw-memory-toggle"><span id={id}>{label}</span>
    <button type="button" role="switch" aria-checked={checked} aria-labelledby={id} className="tw-switch" onClick={() => onChange(!checked)} /></div>;
}

type ViewProps = {
  enabled: boolean; paused?: boolean; memories: Memory[]; loading?: boolean; now?: number; highlight?: string; editing?: Editing | null; busy?: string; menuFor?: string;
  tab?: MemoryTab; query?: string;
  onToggle: (enabled: boolean) => void; onPause?: (paused: boolean) => void; onClearAll: () => void; onEdit: (memory: Memory) => void; onDelete: (memory: Memory) => void;
  onPin?: (memory: Memory) => void; onTab?: (tab: MemoryTab) => void; onQuery?: (query: string) => void;
  onOpenMenu?: (memory: Memory, anchor: MenuAnchor, trigger: HTMLElement) => void;
  onEditChange?: (value: string, truncated: boolean) => void; onEditKind?: (kind: MemoryKind) => void; onEditSave?: () => void; onEditCancel?: () => void;
};

const noop = () => undefined;

/** Presentational page (rendered statically in tests). There is deliberately no add control: the agent records. */
export function MemoryView({ enabled, paused = false, memories, loading = false, now = Date.now(), highlight = '', editing = null, busy = '', menuFor = '', tab = 'all', query = '',
  onToggle, onPause = noop, onClearAll, onPin = noop, onTab = noop, onQuery = noop, onOpenMenu,
  onEditChange = noop, onEditKind = noop, onEditSave = noop, onEditCancel = noop }: ViewProps) {
  // H6-1 / H6-6: title only; 使用记忆 and the reversible 暂停记忆 sit together, 重置记忆 lives apart in the danger zone.
  const head = <><h2 className="tw-memory-title">{MEMORY_COPY.title}</h2>
    <Switch id="tw-memory-toggle-label" label={MEMORY_COPY.toggle} checked={enabled} onChange={onToggle} />
    <Switch id="tw-memory-pause-label" label={MEMORY_COPY.pause} checked={paused} onChange={onPause} /></>;
  if (!memories.length) return <section>{head}{loading ? null : enabled
    ? <div className="tw-empty"><p>{MEMORY_COPY.empty}</p><a className="tw-button" href="/workspace">{MEMORY_COPY.back}</a></div>
    : <div className="tw-empty"><p>{MEMORY_COPY.offEmpty}</p><button type="button" className="tw-button" onClick={() => onToggle(true)}>{MEMORY_COPY.turnOn}</button></div>}</section>;
  const shown = filterMemories(memories, tab, query);
  return <section>{head}
    <div className="tw-memory-tools"><input type="search" className="tw-memory-search" aria-label={MEMORY_COPY.search} placeholder={MEMORY_COPY.search} value={query}
      onChange={event => onQuery(event.target.value)} />
      <div className="tw-memory-tabs" role="tablist" aria-label={MEMORY_COPY.kind}>{MEMORY_TABS.map(([id, label]) =>
        <button key={id} type="button" role="tab" aria-selected={tab === id} className="tw-memory-tab" onClick={() => onTab(id)}>{label}</button>)}</div></div>
    {shown.length ? <ul className="tw-memory-list">{shown.map(memory => <li key={memory.id} data-memory-id={memory.id}
      className={`tw-memory-row${memory.pinned ? ' is-pinned' : ''}${highlight === memory.id ? ' is-highlight' : ''}`}>
      {editing?.id === memory.id
        ? <MemoryEditor value={editing.value} truncated={editing.truncated} kind={editing.kind} busy={busy === memory.id} onChange={onEditChange} onKind={onEditKind} onSave={onEditSave} onCancel={onEditCancel} />
        : <><div className="tw-memory-body"><p className="tw-memory-text">{memory.text}</p>
          <p className="tw-memory-source">{memorySourceHref(memory)
            ? <>来自「<a href={memorySourceHref(memory)!}>{memory.source_task_title}</a>」</> : memorySource(memory)} · <time dateTime={memory.created_at}>{memoryDate(memory.created_at, now)}</time></p>
          <span className="tw-memory-kind">{memoryKindLabel(memory.kind)}</span></div>
          <div className="tw-memory-side"><button type="button" className="tw-memory-pin" aria-pressed={Boolean(memory.pinned)} aria-label={memory.pinned ? MEMORY_COPY.unpin(memory.text) : MEMORY_COPY.pin(memory.text)}
              disabled={busy === memory.id} onClick={() => onPin(memory)}><Pin size={16} aria-hidden="true" /></button>
            <button type="button" className="tw-row-more" aria-haspopup="menu" aria-expanded={menuFor === memory.id} aria-controls={menuFor === memory.id ? MENU_ID : undefined}
              aria-label={MEMORY_COPY.more(memory.text)} disabled={busy === memory.id}
              onClick={event => { const box = event.currentTarget.getBoundingClientRect(); onOpenMenu?.(memory, { left: box.left, top: box.top, right: box.right, bottom: box.bottom }, event.currentTarget); }}>
              <MoreHorizontal size={24} aria-hidden="true" /></button></div></>}
    </li>)}</ul> : <p className="tw-memory-miss">{MEMORY_COPY.noMatch}</p>}
    <div className="tw-memory-danger"><button type="button" className="tw-danger-button" onClick={onClearAll}>{MEMORY_COPY.reset}</button></div>
  </section>;
}

/** 我的 → 记忆. `?settings=memory&memory=<id>` flashes that row (the 已记住 chip links here). */
export function MemorySettings() {
  const [enabled, setEnabled] = useState(true);
  const [paused, setPaused] = useState(false);
  const [memories, setMemories] = useState<Memory[] | null>(null);
  const [error, setError] = useState('');
  const [editing, setEditing] = useState<Editing | null>(null);
  const [busy, setBusy] = useState('');
  const [menu, setMenu] = useState<{ memory: Memory; anchor: MenuAnchor; trigger: HTMLElement } | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [highlight, setHighlight] = useState('');
  const [revision, setRevision] = useState(0);
  const [tab, setTab] = useState<MemoryTab>('all');
  const [query, setQuery] = useState('');
  useEffect(() => {
    const abort = new AbortController();
    void getMemorySettings(abort.signal).then(result => { if (!abort.signal.aborted) { setEnabled(resolveMemoryEnabled(result)); setPaused(result?.paused === true); } }).catch(() => undefined);
    void listMemories(abort.signal).then(found => { if (!abort.signal.aborted) { setMemories(found); setError(''); } })
      .catch(() => { if (!abort.signal.aborted) { setMemories(current => current ?? []); setError(MEMORY_COPY.loadFailed); } });
    return () => abort.abort();
  }, [revision]);
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('memory') || '';
    if (!id || !memories?.some(memory => memory.id === id)) return;
    setHighlight(id);
    document.querySelector(`[data-memory-id="${CSS.escape(id)}"]`)?.scrollIntoView({ block: 'center' });
    const timer = window.setTimeout(() => setHighlight(''), MEMORY_HIGHLIGHT_MS);
    return () => window.clearTimeout(timer);
  }, [memories]);
  async function toggle(next: boolean) {
    const previous = enabled; setEnabled(next);
    try { setEnabled(resolveMemoryEnabled(await setMemoryEnabled(next))); } catch { setEnabled(previous); showToast({ message: MEMORY_COPY.failed }); }
  }
  async function pause(next: boolean) {
    const previous = paused; setPaused(next);
    try { setPaused((await setMemoryPaused(next)).paused === true); } catch { setPaused(previous); showToast({ message: MEMORY_COPY.failed }); }
  }
  const replace = (memory: Memory) => setMemories(current => current?.map(item => item.id === memory.id ? mergeEditedMemory(item, memory) : item) ?? null);
  async function save() {
    if (!editing || !editing.value.trim()) return;
    setBusy(editing.id);
    try {
      const { memory } = await patchMemory(editing.id, { text: editing.value.trim(), kind: editing.kind });
      replace(memory); setEditing(null);
    } catch { showToast({ message: MEMORY_COPY.failed }); } finally { setBusy(''); }
  }
  async function pin(memory: Memory) {
    setBusy(memory.id);
    try { replace((await patchMemory(memory.id, { pinned: !memory.pinned })).memory); } catch { showToast({ message: MEMORY_COPY.failed }); } finally { setBusy(''); }
  }
  async function remove(memory: Memory) {
    setBusy(memory.id);
    try {
      await deleteMemory(memory.id);
      setMemories(current => current?.filter(item => item.id !== memory.id) ?? null);
      showToast({ message: MEMORY_COPY.deleted, action: { label: MEMORY_COPY.undo,
        run: () => void restoreMemory(memory.id).then(() => setRevision(value => value + 1)).catch(() => showToast({ message: MEMORY_COPY.failed })) } });
    } catch { showToast({ message: MEMORY_COPY.failed }); } finally { setBusy(''); }
  }
  async function clearAll() {
    setConfirming(false);
    try { await clearMemories(); setMemories([]); setEditing(null); } catch { showToast({ message: MEMORY_COPY.failed }); }
  }
  const startEdit = (memory: Memory) => setEditing({ id: memory.id, value: memory.text, truncated: false, kind: memory.kind ?? 'other' });
  return <>
    <MemoryView enabled={enabled} paused={paused} memories={memories ?? []} loading={memories === null} highlight={highlight} editing={editing} busy={busy} menuFor={menu?.memory.id}
      tab={tab} query={query} onTab={setTab} onQuery={setQuery} onPin={memory => void pin(memory)}
      onToggle={next => void toggle(next)} onPause={next => void pause(next)} onClearAll={() => setConfirming(true)}
      onEdit={startEdit} onDelete={memory => void remove(memory)}
      onOpenMenu={(memory, anchor, trigger) => setMenu({ memory, anchor, trigger })}
      onEditChange={(value, truncated) => setEditing(current => current && { ...current, value, truncated })}
      onEditKind={kind => setEditing(current => current && { ...current, kind })}
      onEditSave={() => void save()} onEditCancel={() => setEditing(null)} />
    {error && <p role="alert" className="tw-error">{error}</p>}
    {menu && <PopMenu id={MENU_ID} label={MEMORY_COPY.more(menu.memory.text)} anchor={menu.anchor} returnFocus={menu.trigger} onClose={() => setMenu(null)}>
      <MemoryRowMenu onEdit={() => { setMenu(null); startEdit(menu.memory); }}
        onDelete={() => { setMenu(null); void remove(menu.memory); }} /></PopMenu>}
    {confirming && memories && <ClearAllDialog onCancel={() => setConfirming(false)} onConfirm={() => void clearAll()} />}
    <ExperienceSection />
  </>;
}

/** Phase 2 我的经验 (presentational): newest first, one delete per row, and the anonymous-statistics switch. */
export function ExperienceView({ experiences, contribute, loading = false, busy = '', now = Date.now(), onContribute, onDelete }: {
  experiences: Experience[]; contribute: boolean; loading?: boolean; busy?: string; now?: number;
  onContribute: (contribute: boolean) => void; onDelete: (experience: Experience) => void;
}) {
  return <section className="tw-experience" aria-labelledby="tw-experience-title" data-testid="tw-experience">
    <h2 id="tw-experience-title" className="tw-memory-title">{EXPERIENCE_COPY.title}</h2>
    <p className="tw-muted">{EXPERIENCE_COPY.line}</p>
    <Switch id="tw-experience-contribute-label" label={EXPERIENCE_COPY.contribute} checked={contribute} onChange={onContribute} />
    <p className="tw-muted">{EXPERIENCE_COPY.contributeLine}</p>
    {loading ? null : experiences.length ? <ul className="tw-memory-list">{experiences.map(experience =>
      <li key={experience.id} className="tw-memory-row" data-experience-id={experience.id}>
        <div className="tw-memory-body"><p className="tw-memory-text">{experience.title}</p>
          <p className="tw-memory-source">{experience.summary}</p>
          <p className="tw-memory-source"><time dateTime={experience.created_at}>{memoryDate(experience.created_at, now)}</time>
            {experience.success_count > 0 ? ` · ${EXPERIENCE_COPY.reused(experience.success_count)}` : ''}</p></div>
        <div className="tw-memory-side"><button type="button" className="tw-row-more" aria-label={EXPERIENCE_COPY.remove(experience.title)}
          disabled={busy === experience.id} onClick={() => onDelete(experience)}><Trash2 size={16} aria-hidden="true" /></button></div>
      </li>)}</ul> : <p className="tw-memory-miss">{EXPERIENCE_COPY.empty}</p>}
  </section>;
}

export function ExperienceSection() {
  const [experiences, setExperiences] = useState<Experience[] | null>(null);
  const [contribute, setContribute] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    const abort = new AbortController();
    void listExperiences(abort.signal).then(found => { if (!abort.signal.aborted) { setExperiences(found.experiences); setContribute(found.settings.contribute_global); } })
      .catch(() => { if (!abort.signal.aborted) { setExperiences([]); setError(EXPERIENCE_COPY.loadFailed); } });
    return () => abort.abort();
  }, []);
  async function toggle(next: boolean) {
    const previous = contribute; setContribute(next);
    try { setContribute((await setExperienceContribution(next)).contribute_global !== false); } catch { setContribute(previous); showToast({ message: EXPERIENCE_COPY.failed }); }
  }
  async function remove(experience: Experience) {
    setBusy(experience.id);
    try {
      await deleteExperience(experience.id);
      setExperiences(current => current?.filter(item => item.id !== experience.id) ?? null);
      showToast({ message: EXPERIENCE_COPY.deleted });
    } catch { showToast({ message: EXPERIENCE_COPY.failed }); } finally { setBusy(''); }
  }
  return <>
    <ExperienceView experiences={experiences ?? []} loading={experiences === null} contribute={contribute} busy={busy}
      onContribute={next => void toggle(next)} onDelete={experience => void remove(experience)} />
    {error && <p role="alert" className="tw-error">{error}</p>}
  </>;
}

/** H6-7: 「仅用本项目记忆」 — on ⇒ this project's conversations get no account-wide memory. */
export function ProjectMemoryToggleView({ projectOnly, onToggle }: { projectOnly: boolean; onToggle: (projectOnly: boolean) => void }) {
  return <Switch id="tw-project-memory-label" label={MEMORY_COPY.projectOnly} checked={projectOnly} onChange={onToggle} />;
}

export function ProjectMemoryToggle({ projectId }: { projectId: string }) {
  const [projectOnly, setProjectOnly] = useState(false);
  useEffect(() => {
    const abort = new AbortController();
    void getProjectMemorySettings(projectId, abort.signal).then(result => { if (!abort.signal.aborted) setProjectOnly(result.project_only === true); }).catch(() => undefined);
    return () => abort.abort();
  }, [projectId]);
  async function toggle(next: boolean) {
    const previous = projectOnly; setProjectOnly(next);
    try { setProjectOnly((await setProjectMemoryOnly(projectId, next)).project_only === true); } catch { setProjectOnly(previous); showToast({ message: MEMORY_COPY.failed }); }
  }
  return <ProjectMemoryToggleView projectOnly={projectOnly} onToggle={next => void toggle(next)} />;
}
