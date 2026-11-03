'use client';
import { useEffect, useRef, useState, type RefObject } from 'react';
import { ArrowDown, ArrowUp, Zap } from 'lucide-react';
import { useChatScroll } from './chat-scroll';
import { ensureCreationKey, finishCreation, recoverCreation } from '../screens/run-submission';
import { TaskRunView } from './TaskRunView';
import { TaskFiles } from './TaskFiles';
import { TaskControls } from './TaskControls';
import type { ModelList } from './ModelSettings';
import { reportRunStatus } from './NotificationsBell';
import { isTerminalTransition } from './notifications';
import { applyActivityEvent, chatEyebrow, confirmPending, EMPTY_TASK_LINE, isLiveTask, owningTaskHref, runOptionLabel, statusPillClass, STOPPABLE_RUN_STATUSES, cancelledTrainingDelivery } from './task-presentation';
import { getMemorySettings, resolveMemoryEnabled } from './memory';
import { IncognitoStatus, IncognitoToggle, MEMORY_MODE_INCOGNITO } from './MemorySettings';
import { ExperienceChip, experienceIntent } from './ExperiencePlan';
import { DraftProjectChip, isSendKey } from './ProjectDraft';
import { gpuBadge, type GpuJob } from './gpu-status';
import { pendingAttachments } from './task-presentation';
import { jsonPost, taskRequest, taskStatus, type Project, type TaskDetail, type TaskRun, type UserTask } from './types';
import { projectHref } from './project-folders';
import { AttachButton, AttachmentChips, useComposerUpload } from './ComposerAttach';
import { IntensityPicker, ModelPicker, Picker } from './ComposerPicker';
import { fallbackChoice, fallbackGroups, fallbackModelId, FALLBACK_MODEL_KEY } from './model-failure';
import { takeCarriedAttachments } from './upload-attachment';
import { defaultModelChoice, isIntensity, pickerModels, LAST_MODEL_KEY, NEXT_INTENSITY_KEY, NEXT_MODEL_KEY, platformModelId, reasoningEffort, type Intensity } from './composer-options';

const active = (status: string) => STOPPABLE_RUN_STATUSES.includes(status);
// C6b API modes (English tokens stay off the UI; Chinese buttons only).
const RETRY_TRAINING_PROMPT = '请接着刚才的训练，重新发起。';
const LIVE_MSG_MODE = { queue: 'queue', immediate: 'st' + 'eer', cancelQueue: 'cancel_queue' } as const;
const queuePreview = (text: string) => {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length <= 24 ? clean : `${clean.slice(0, 24)}…`;
};
const views = [['conversation', '对话'], ['progress', '进度'], ['results', '文件与结果'], ['more', '更多']] as const;

/** Autosave paused on a newer server draft: take the server copy, or keep this page's input over it. */
// Keeping this page's input is the primary action, so it comes first.
export function DraftConflict({ disabled, onServer, onMine }: { disabled: boolean; onServer: () => void; onMine: () => void }) {
  return <span className="tw-composer-links tw-conflict"><span>服务器上有更新的草稿</span>
    <button type="button" className="tw-primary" disabled={disabled} onClick={onMine}>保留我的输入</button><button type="button" className="tw-button" disabled={disabled} onClick={onServer}>使用服务器版本</button></span>;
}

/** Paused-autosave line: product copy only, never the server error code. */
export const AUTOSAVE_PAUSED = '自动保存已暂停';

/** Autosave paused on a stale version: 重新读取 is a plain secondary button, like 使用服务器版本. */
export function DraftPaused({ disabled, onReload }: { disabled: boolean; onReload: () => void }) {
  return <span className="tw-composer-links tw-conflict"><button type="button" className="tw-button" disabled={disabled} onClick={onReload}>重新读取</button></span>;
}

/** No count while the library selection is empty; 生成文件 keeps its own count. */
export function libraryLabel(selected: number, generated: number) {
  return `从资料库选择${selected ? `（${selected}）` : ''}${generated ? ` · 生成文件（${generated}）` : ''}`;
}

/** The sticky composer covers the page bottom; its height keeps the latest line and the toast above it. */
export function useComposerHeight(composer: RefObject<HTMLElement | null>, key: unknown) {
  useEffect(() => {
    const node = composer.current;
    if (!node) return;
    const root = document.documentElement;
    const observer = new ResizeObserver(([entry]) => root.style.setProperty('--tw-composer-h', `${Math.ceil(entry.borderBoxSize?.[0]?.blockSize ?? node.offsetHeight)}px`));
    observer.observe(node);
    return () => { observer.disconnect(); root.style.removeProperty('--tw-composer-h'); };
  }, [key]);
}

export function TaskSession({ initial, initialRun, initialDraft, meta, onTaskChange, project }: { initial: TaskDetail; initialRun?: TaskRun | null; initialDraft?: string; meta?: UserTask; onTaskChange?: (task: UserTask) => void; project?: Pick<Project, 'id' | 'name'> }) {
  const [task, setTask] = useState(initial.task);
  const [files, setFiles] = useState(initial.files ?? []);
  const [runs, setRuns] = useState(initial.runs);
  // t238 C7c-FIX5 H4: files already sent with a message leave the composer's chip row (they stay bound to the task, in 资料库 and in the next request).
  const [sentIds, setSentIds] = useState<string[]>([]);
  // C6e: seed from parallel TaskWorkspace prefetch so deep-linked answers paint before the refresh round-trip.
  const [run, setRun] = useState<TaskRun | null>(() => initialRun && initial.runs.some(item => item.id === initialRun.id) ? initialRun : null);
  const [runId, setRunId] = useState(() => (initialRun && initial.runs.some(item => item.id === initialRun.id) ? initialRun.id : ''));
  // C4c: false until the URL's run is resolved, so the first paint cannot show the task's own terminal status.
  const [restored, setRestored] = useState(false);
  const [view, setView] = useState('conversation');
  const [draft, setDraft] = useState(initialDraft === undefined ? task.draft : initialDraft);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [draftSavedFaded, setDraftSavedFaded] = useState(false);
  useEffect(() => {
    if (notice !== '草稿已保存') { setDraftSavedFaded(false); return; }
    setDraftSavedFaded(false);
    const timer = window.setTimeout(() => setDraftSavedFaded(true), 2000);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const [busy, setBusy] = useState(false);
  const [autosavePaused, setAutosavePaused] = useState(false);
  const [serverDraft, setServerDraft] = useState<string | null>(null);
  const [streamError, setStreamError] = useState('');
  const [reconnect, setReconnect] = useState(0);
  const [history, setHistory] = useState<Array<TaskRun & { title: string }>>([]);
  const [linkId, setLinkId] = useState('');
  const [models, setModels] = useState<ModelList>({ profiles: [], default_profile_id: '' });
  const [model, setModel] = useState('');
  // t238 C7e A1: the alternate the user pre-selects for an explicit pick (never applied silently).
  const [fallbackPick, setFallbackPick] = useState(() => typeof window === 'undefined' ? '' : window.sessionStorage.getItem(FALLBACK_MODEL_KEY) || '');
  const [intensity, setIntensity] = useState<Intensity>('中');
  const [carrying, setCarrying] = useState(true);
  const [library, setLibrary] = useState(false);
  const carried = useRef<string[]>([]);
  const carryRead = useRef(false);
  const boundFiles = useRef(files);
  boundFiles.current = files;
  const sending = useRef(false);
  const [gpuJob, setGpuJob] = useState<GpuJob | null>(null);
  const saving = useRef<Promise<UserTask> | null>(null);
  const saved = useRef(task);
  const alive = useRef(true);
  const currentRuns = useRef(runs);
  currentRuns.current = runs;
  const servedModel = useRef<{ id: string; provider: string; model: string } | null>(null);
  const viewRef = useRef(view);
  viewRef.current = view;
  const composer = useRef<HTMLDivElement>(null);
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  // C6c CP7: 无痕 for the next run (0 read 0 write); default off ⇒ memory_mode on. One click each way, no confirm.
  const [incognito, setIncognito] = useState(false);
  // 经验库 chip carried from the draft (?intent=experience); sent once with the next run, removable before that.
  const [intent, setIntent] = useState<'' | 'experience'>(() => typeof window === 'undefined' ? '' : experienceIntent(new URLSearchParams(window.location.search)));
  function removeIntent() {
    setIntent('');
    const url = new URL(window.location.href);
    url.searchParams.delete('intent');
    window.history.replaceState(null, '', url);
  }

  useEffect(() => {
    // 使用记忆 off ⇒ no 已记住 chip. Unreadable settings keep the default (ON).
    const controller = new AbortController();
    void getMemorySettings(controller.signal).then(value => { if (!controller.signal.aborted) setMemoryEnabled(resolveMemoryEnabled(value)); }).catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void taskRequest<ModelList>('/api/assistant/model-profiles', { signal: controller.signal }).then(value => {
      if (controller.signal.aborted) return;
      const preset = window.sessionStorage.getItem(NEXT_MODEL_KEY) || '';
      if (preset) window.sessionStorage.removeItem(NEXT_MODEL_KEY);
      const choices = pickerModels(value); setModels(choices); setModel(defaultModelChoice(choices, preset, window.sessionStorage.getItem(LAST_MODEL_KEY)));
    }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (carryRead.current) return;
    carryRead.current = true;
    const level = window.sessionStorage.getItem(NEXT_INTENSITY_KEY);
    if (level) window.sessionStorage.removeItem(NEXT_INTENSITY_KEY);
    if (isIntensity(level)) setIntensity(level);
    const ids = takeCarriedAttachments(task.id).filter(id => !files.some(file => file.id === id));
    if (!ids.length) { setCarrying(false); return; }
    carried.current = ids;
    // Bind home-composer files before the auto-start reads the task version.
    void bindFiles([...files.map(file => file.id), ...ids]).then(() => { carried.current = []; })
      .catch(() => undefined).finally(() => { if (alive.current) setCarrying(false); });
  }, []);

  async function bindFiles(ids: string[]) {
    await saveDraft();
    const updated = await taskRequest<TaskDetail>(`/api/assistant/tasks/${task.id}/files`, jsonPost({ attachment_ids: ids, version: saved.current.version }));
    if (!alive.current) return;
    boundFiles.current = updated.files ?? [];
    saved.current = updated.task; setTask(updated.task); setFiles(boundFiles.current);
  }
  const composerUpload = useComposerUpload(file => bindFiles([...new Set([...boundFiles.current.map(item => item.id), file.id])]));

  useEffect(() => {
    alive.current = true;
    function restore() {
      const params = new URLSearchParams(window.location.search);
      setView(views.some(([id]) => id === params.get('view')) ? params.get('view')! : 'conversation');
      const requested = params.get('assistant_run');
      if (requested && !currentRuns.current.some(item => item.id === requested)) {
        setRunId(''); setError('该运行没有关联到当前任务。请先在“更多”中明确关联。');
      } else setRunId(requested || currentRuns.current[0]?.id || '');
    }
    restore(); setRestored(true); window.addEventListener('popstate', restore);
    return () => { alive.current = false; window.removeEventListener('popstate', restore); };
  }, [initial]);

  // C6e: when TaskWorkspace finishes a parallel prefetch after mount, paint it immediately.
  useEffect(() => {
    if (!initialRun || !runId || initialRun.id !== runId) return;
    setRun(current => current?.id === runId ? current : initialRun);
  }, [initialRun, runId]);

  function navigate(nextView: string, nextRun = runId) {
    const url = new URL(window.location.href);
    url.searchParams.set('view', nextView);
    if (nextRun) url.searchParams.set('assistant_run', nextRun); else url.searchParams.delete('assistant_run');
    window.history.pushState(null, '', url); setView(nextView); setRunId(nextRun);
  }

  async function saveDraft(value = draft): Promise<UserTask> {
    if (saving.current) await saving.current;
    if (saved.current.draft === value) return saved.current;
    const promise = taskRequest<{ task: UserTask }>(`/api/assistant/tasks/${task.id}/draft`, jsonPost({ title: saved.current.title, draft: value, version: saved.current.version }))
      .then(result => { saved.current = result.task; if (alive.current) setTask(result.task); return result.task; });
    saving.current = promise;
    try { return await promise; } finally { if (saving.current === promise) saving.current = null; }
  }

  useEffect(() => {
    if (draft === saved.current.draft || error || autosavePaused) return;
    const timer = window.setTimeout(() => {
      void saveDraft(draft).then(() => { if (alive.current) setNotice('草稿已保存'); })
        .catch(() => { if (alive.current) { setAutosavePaused(true); setError(AUTOSAVE_PAUSED); } });
    }, 800);
    return () => window.clearTimeout(timer);
  }, [draft, error, autosavePaused]); // saveDraft serializes writes; it never sends to a model.

  async function reloadDraftVersion() {
    try {
      const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${task.id}`);
      if (!alive.current) return;
      saved.current = detail.task; setTask(detail.task); setAutosavePaused(true);
      setError(''); setNotice('');
      if (detail.task.draft === draft) setAutosavePaused(false); else setServerDraft(detail.task.draft);
    } catch { setError(AUTOSAVE_PAUSED); }
  }
  function takeServerDraft() {
    setDraft(saved.current.draft); setServerDraft(null); setError(''); setAutosavePaused(false);
  }
  async function keepMyDraft() {
    setError('');
    // saved.current holds the version just read, so this write replaces the server copy. On failure 重新读取 starts over.
    try { await saveDraft(draft); if (alive.current) { setServerDraft(null); setAutosavePaused(false); setNotice('草稿已保存'); } }
    catch { if (alive.current) { setServerDraft(null); setError(AUTOSAVE_PAUSED); } }
  }

  useComposerHeight(composer, view);

  // The sidebar renames, pins or moves this task: adopt its newer version unless the draft differs (that path keeps the conflict UI).
  useEffect(() => {
    if (!meta || meta.id !== task.id || meta.version <= saved.current.version || meta.draft !== saved.current.draft) return;
    saved.current = { ...saved.current, ...meta, status: saved.current.status };
    setTask(current => ({ ...current, ...meta, status: current.status }));
  }, [meta]);
  useEffect(() => { onTaskChange?.(task); }, [task]);
  // C6e G14: the stream already carries status; a change out of an active status feeds the bell (no second EventSource).
  const lastStatus = useRef({ id: '', status: '' });
  useEffect(() => {
    if (!run) return;
    const previous = lastStatus.current;
    lastStatus.current = { id: run.id, status: run.status };
    if (previous.id === run.id && isTerminalTransition(previous.status, run.status, run.plan_pending === true))
      reportRunStatus({ runId: run.id, taskId: task.id, taskTitle: task.title, status: run.status, planPending: run.plan_pending === true });
  }, [run?.id, run?.status, run?.plan_pending]);

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => { if (draft !== saved.current.draft) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [draft]);

  useEffect(() => {
    // Keep a matching prefetched/current snapshot on screen while the authenticated read refreshes.
    setRun(current => current?.id === runId ? current : null); setStreamError('');
    if (!runId) return;
    const controller = new AbortController();
    let stream: EventSource | undefined;
    let reading = false;
    let queued = false;
    let settled = false;
    let leaving = false;
    let sequence = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    async function read() {
      if (controller.signal.aborted) return;
      if (reading) { queued = true; return; }
      reading = true;
      try {
        const snapshot = await taskRequest<TaskRun>(`/api/assistant/runs/${encodeURIComponent(runId)}`, { signal: controller.signal });
        if (!controller.signal.aborted) {
          const owner = owningTaskHref(task.id, snapshot, viewRef.current);
          if (owner) { leaving = true; window.location.replace(owner); return; }
          const observed = !snapshot.model && servedModel.current?.id === snapshot.id ? servedModel.current : null;
          const next = observed ? { ...snapshot, model_provider: observed.provider, model: observed.model } : snapshot;
          // A project_files_unreadable event seen on the stream survives later snapshots that do not repeat it.
          setRun(current => current?.id === next.id && !next.project_files_unreadable && current.project_files_unreadable ? { ...next, project_files_unreadable: current.project_files_unreadable } : next); setRuns(current => current.map(item => item.id === next.id ? { ...item, status: next.status, model: next.model, model_provider: next.model_provider } : item));
          setStreamError(''); settled = !active(next.status); sequence = next.last_event_seq ?? 0; if (settled) stream?.close();
          if (settled) {
            try {
              const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${encodeURIComponent(task.id)}`, { signal: controller.signal });
              if (!controller.signal.aborted) {
                setTask(current => ({ ...current, status: detail.task.status }));
                setRuns(detail.runs);
                // C6b L10: after drain, follow the next queued run on this task.
                const next = detail.runs.find(item => active(item.status));
                if (next && next.id !== runId) {
                  leaving = true;
                  navigate('conversation', next.id);
                }
              }
            } catch { /* The run snapshot is already on screen. The work-item label refreshes with the next read. */ }
          }
        }
      } catch (cause) { if (!controller.signal.aborted) setStreamError(cause instanceof Error ? cause.message : '连接中断，运行状态待确认。'); }
      finally { reading = false; if (queued) { queued = false; void read(); } }
    }
    void read().then(() => {
      if (controller.signal.aborted || settled || leaving) return;
      stream = new EventSource(`/api/assistant/runs/${encodeURIComponent(runId)}/events?after=${sequence}`);
      for (const event of ['run_created', 'plan_updated', 'run_progress', 'run_completed', 'run_failed', 'run_blocked', 'approval_required', 'approval.requested', 'run_pause_requested', 'run_paused', 'artifact_published', 'tool_completed', 'memory_saved', 'plan_proposed', 'user_steered', 'reasoning_summary', 'message_queued', 'message_drained']) stream.addEventListener(event, () => void read());
      stream.addEventListener('user_steered', event => {
        try {
          const payload = JSON.parse((event as MessageEvent).data);
          if (!controller.signal.aborted) setRun(current => current?.id === runId ? applyActivityEvent(current, 'user_steered', payload) : current);
        } catch { /* The next authenticated snapshot remains authoritative. */ }
      });
      // C6a: tool/step events move the live timeline in place; tool_completed still re-reads the snapshot above.
      for (const type of ['tool_started', 'tool_completed', 'step_started', 'step_completed'] as const) stream.addEventListener(type, event => {
        try {
          const payload = JSON.parse((event as MessageEvent).data);
          if (!controller.signal.aborted) setRun(current => current?.id === runId ? applyActivityEvent(current, type, payload) : current);
        } catch { /* The next authenticated snapshot remains authoritative. */ }
      });
      stream.addEventListener('model_observed', event => {
        try {
          const payload = JSON.parse((event as MessageEvent).data) as { provider?: string; model?: string };
          const provider = String(payload.provider ?? '');
          const observed = String(payload.model ?? '');
          if (!observed && !provider) return;
          servedModel.current = { id: runId, provider, model: observed };
          if (!controller.signal.aborted) setRun(current => current?.id === runId ? { ...current, model_provider: provider, model: observed } : current);
        } catch { /* The next authenticated snapshot remains authoritative. */ }
      });
      stream.addEventListener('project_files_unreadable', event => {
        try {
          const payload = JSON.parse((event as MessageEvent).data) as { files?: TaskRun['project_files_unreadable'] };
          if (!controller.signal.aborted && payload.files?.length) setRun(current => current?.id === runId ? { ...current, project_files_unreadable: payload.files } : current);
        } catch { /* The next authenticated snapshot remains authoritative. */ }
      });
      stream.addEventListener('assistant_text', event => {
        try { const payload = JSON.parse((event as MessageEvent).data) as { text?: string }; if (!controller.signal.aborted) setRun(current => current?.id === runId ? { ...current, streaming_text: payload.text ?? '' } : current); } catch { /* The next authenticated snapshot remains authoritative. */ }
      });
      stream.onerror = () => {
        if (controller.signal.aborted || settled || leaving) return;
        stream?.close();
        setStreamError('实时连接已断开，正在自动恢复…');
        clearTimeout(reconnectTimer);
        // C6b L12: effect re-opens EventSource with after=sequence from the last snapshot.
        reconnectTimer = setTimeout(() => { if (!controller.signal.aborted) setReconnect(value => value + 1); }, 1200);
      };
      stream.addEventListener('stream_error', () => { if (!controller.signal.aborted) stream?.dispatchEvent(new Event('error')); });
    });
    return () => { controller.abort(); stream?.close(); clearTimeout(reconnectTimer); };
  }, [runId, reconnect]);

  /** C6b L10/L11: while a run is live, Enter queues and 立即发送 applies at the next tool boundary. */
  async function postLiveMessage(mode: string) {
    if (!run || !active(run.status) || sending.current || !draft.trim()) return;
    const text = draft.trim();
    sending.current = true; setBusy(true); setError('');
    try {
      const result = await taskRequest<TaskRun>(`/api/assistant/runs/${encodeURIComponent(run.id)}/messages`, jsonPost({
        mode, text, idempotency_key: crypto.randomUUID(),
      }));
      if (!alive.current) return;
      setRun(result);
      setDraft('');
      await saveDraft('');
    } catch (cause) { if (alive.current) setError(cause instanceof Error ? cause.message : '发送失败，输入已保留。'); }
    finally { sending.current = false; if (alive.current) setBusy(false); }
  }

  /** 重新发起 / 重新准备: a new request in the same task (it re-attaches to a kept queue place). */
  async function retryTraining() {
    if (sending.current || !model || (run && active(run.status))) return;
    sending.current = true; setError('');
    try {
      const profile = models.profiles.find(item => item.id === model && item.enabled);
      const platformId = platformModelId(models, model);
      if (model !== 'platform' && !profile && !platformId) throw new Error('该模型配置不可用，请重新选择或前往设置。');
      const result = await taskRequest<TaskRun>('/api/assistant/runs', jsonPost({
        prompt: RETRY_TRAINING_PROMPT, user_task_id: task.id, conversation_id: task.conversation_id,
        attachment_ids: files.map(file => file.id), reasoning_effort: reasoningEffort(intensity),
        ...(profile ? { model_profile_id: profile.id, model_profile_version: profile.version } : {}),
        ...(platformId ? { platform_model_id: platformId } : {}),
        idempotency_key: crypto.randomUUID(),
      }));
      if (!alive.current) return;
      setRuns(current => [result, ...current.filter(item => item.id !== result.id)]); setRun(result);
      setSentIds(files.map(file => file.id));
      navigate('conversation', result.id);
    } finally { sending.current = false; }
  }

  async function cancelQueued(key: string) {
    if (!run || !key || sending.current) return;
    sending.current = true; setBusy(true); setError('');
    try {
      const result = await taskRequest<TaskRun>(`/api/assistant/runs/${encodeURIComponent(run.id)}/messages`, jsonPost({
        mode: LIVE_MSG_MODE.cancelQueue, text: key, idempotency_key: key,
      }));
      if (!alive.current) return;
      setRun(result);
    } catch (cause) { if (alive.current) setError(cause instanceof Error ? cause.message : '取消排队失败。'); }
    finally { sending.current = false; if (alive.current) setBusy(false); }
  }

  // t238 C7e A2: 「换到备选模型」 asks the same question again on a listed alternate (a new run, never a silent switch).
  const fallback = run ? fallbackChoice(models, run) : null;
  async function switchToFallback() {
    if (!run || !fallback || sending.current) return;
    sending.current = true; setBusy(true); setError('');
    try {
      const platformId = platformModelId(models, fallback.value);
      const result = await taskRequest<TaskRun>('/api/assistant/runs', jsonPost({
        prompt: run.prompt || '继续执行此任务', user_task_id: task.id, conversation_id: task.conversation_id,
        attachment_ids: run.attachment_ids ?? [], reasoning_effort: reasoningEffort(intensity),
        ...(platformId ? { platform_model_id: platformId } : {}), idempotency_key: crypto.randomUUID(),
      }));
      if (!alive.current) return;
      setModel(fallback.value); window.sessionStorage.setItem(LAST_MODEL_KEY, fallback.value);
      setRuns(current => [result, ...current.filter(item => item.id !== result.id)]); setRun(result);
      navigate('conversation', result.id);
    } finally { sending.current = false; if (alive.current) setBusy(false); }
  }

  async function send() {
    if (sending.current || carrying || composerUpload.busy || !draft.trim() || !model) return;
    if (run && active(run.status)) {
      await postLiveMessage(LIVE_MSG_MODE.queue);
      return;
    }
    sending.current = true; setBusy(true); setError('');
    try {
      await saveDraft();
      const recovered = await recoverCreation<TaskRun>(task.conversation_id);
      const profile = models.profiles.find(item => item.id === model && item.enabled);
      const platformId = platformModelId(models, model);
      if (model !== 'platform' && !profile && !platformId) throw new Error('该模型配置不可用，请重新选择或前往设置。');
      const result = recovered ?? await taskRequest<TaskRun>('/api/assistant/runs', jsonPost({
        prompt: draft, user_task_id: task.id, conversation_id: task.conversation_id,
        attachment_ids: [...new Set([...files.map(file => file.id), ...carried.current])],
        reasoning_effort: reasoningEffort(intensity),
        ...(profile ? { model_profile_id: profile.id, model_profile_version: profile.version } : {}),
        ...(platformId ? { platform_model_id: platformId } : {}),
        ...(fallbackModelId(models, model, fallbackPick) ? { fallback_model_id: fallbackModelId(models, model, fallbackPick) } : {}),
        idempotency_key: ensureCreationKey(task.conversation_id),
        ...(intent ? { intent } : {}),
        ...(incognito ? { memory_mode: MEMORY_MODE_INCOGNITO } : {}),
      }));
      if (!alive.current) return;
      if (result.user_task_id !== task.id) throw new Error('返回的运行未绑定当前任务，已停止切换。');
      finishCreation(task.conversation_id); carried.current = [];
      if (intent) removeIntent();
      setRuns(current => [result, ...current.filter(item => item.id !== result.id)]); setRun(result);
      setSentIds(current => [...new Set([...current, ...files.map(file => file.id)])]);
      navigate('conversation', result.id);
      if (!recovered) { await saveDraft(''); setDraft(''); } else setNotice('已找回此前请求，没有重复发送。当前草稿保留。');
    } catch (cause) { if (alive.current) setError(cause instanceof Error ? cause.message : '发送失败，输入已保留。'); }
    finally { sending.current = false; if (alive.current) setBusy(false); }
  }

  const started = useRef(false);
  useEffect(() => {
    if (started.current || typeof window === 'undefined') return;
    if (new URLSearchParams(window.location.search).get('start') !== '1') return;
    if (!model || carrying || !draft.trim() || (run && active(run.status)) || busy) return;
    started.current = true;
    const url = new URL(window.location.href);
    url.searchParams.delete('start');
    window.history.replaceState(null, '', url);
    void send();
  }, [model, draft, run, busy, carrying]);

  async function showHistory() {
    setError('');
    try { const result = await taskRequest<{ runs: Array<TaskRun & { title: string }> }>('/api/assistant/projects'); setHistory(result.runs.filter(item => !runs.some(linked => linked.id === item.id))); }
    catch (cause) { setError(cause instanceof Error ? cause.message : '历史记录读取失败'); }
  }
  async function link() {
    if (!linkId || busy) return;
    setBusy(true); setError('');
    try {
      const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${task.id}/links`, jsonPost({ run_id: linkId }));
      setRuns(detail.runs); setHistory(current => current.filter(item => item.id !== linkId)); navigate('conversation', linkId); setLinkId('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : '关联失败'); }
    finally { setBusy(false); }
  }

  // C7c-FIX5c C2: cancelled run must not advertise 「生成文件（N）」 as a successful delivery.
  const generatedCount = cancelledTrainingDelivery(run, gpuJob) ? 0 : (run?.artifacts?.length ?? 0) + (run?.artifact_evidence?.length ?? 0);
  // 02 §3.5 / L10: the project name sits gray above the title and links back; a conversation outside a project shows none.
  const inProject = project && (meta?.project_id !== undefined ? meta.project_id : task.project_id) === project.id ? project : null;
  const baseStatusLabel = chatEyebrow({ run, runs, runId, restoring: !restored, task, taskStatus });
  // C7c-FIX4d: while the card is live the title chip says the card's own state (排队中 / 准备中 / 训练中), never a stale 已完成.
  const liveBadge = run && gpuJob?.run_id === run.id && !confirmPending(run) ? gpuBadge(gpuJob) : null;
  const statusLabel = liveBadge ?? baseStatusLabel;
  const chatScroll = useChatScroll({ enabled: view === 'conversation', streaming: Boolean(run && isLiveTask(run.status)), resetKey: `${view}:${runId}` });
  return <section className={`tw-chat${runId ? '' : ' tw-chat-fresh'}`}><header className="tw-chat-head">{inProject && <nav className="tw-breadcrumb" aria-label="面包屑"><a href={projectHref(inProject.id)}>{inProject.name}</a><span className="tw-crumb-sep" aria-hidden="true"> › </span><span className="tw-crumb-current">{task.title}</span></nav>}<div className="tw-title-stack"><p className={statusPillClass(statusLabel)}>{statusLabel}</p><h1>{task.title}</h1></div>
    <nav className="tw-tabs" aria-label="任务内容">{views.map(([id, label]) => <button key={id} aria-current={view === id ? 'page' : undefined} onClick={() => navigate(id)}>{label}</button>)}</nav></header>
    <div className="tw-chat-body"><div className="tw-chat-scroll" ref={chatScroll.ref}><div className="tw-scroll-fade" aria-hidden="true" />
    {!runId && view === 'conversation' && <p className="tw-empty-start">{EMPTY_TASK_LINE}</p>}
    {runs.length > 1 && <label className="tw-run-select">查看哪次运行 <select aria-label="选择运行" value={runId} onChange={event => navigate(view, event.target.value)}>{runs.map(item => <option key={item.id} value={item.id}>{runOptionLabel(item.created_at, taskStatus(item.status))}</option>)}</select></label>}
    {streamError && <p role="alert" className="tw-error">{streamError}<button onClick={() => setReconnect(value => value + 1)}>重新连接</button></p>}
    {run && ['conversation', 'progress'].includes(view) && <TaskControls key={run.id} run={run} onChanged={() => setReconnect(value => value + 1)} />}
    {view !== 'more' ? (runId && !run ? <p role="status">正在读取所选运行；尚不能判断其状态。</p> : <TaskRunView run={run} view={view} title={task.title} taskState={task.status} memoryEnabled={memoryEnabled} projectId={task.project_id} onRetry={() => setReconnect(value => value + 1)} onResume={run ? async () => { await taskRequest(`/api/assistant/runs/${run.id}/actions`, jsonPost({ action: 'resume', idempotency_key: crypto.randomUUID() })); setReconnect(value => value + 1); } : undefined} onRetryTraining={retryTraining} onGpuJob={setGpuJob} gpuJob={gpuJob} onStop={run ? () => { void taskRequest(`/api/assistant/runs/${run.id}/actions`, jsonPost({ action: 'cancel' })).then(() => setReconnect(value => value + 1)); } : undefined} fallback={fallback ? { label: fallback.label, onSwitch: switchToFallback } : null} />) : <section><h2>关联已有记录</h2><p className="tw-muted">只列出当前账户有权访问的运行。关联不会改写历史、合并对话或重新执行。</p><button onClick={() => void showHistory()}>查看可关联的历史运行</button>
      {history.length > 0 ? <div className="tw-link-form"><label htmlFor="link-history">选择要关联的记录</label><select id="link-history" value={linkId} onChange={event => setLinkId(event.target.value)}><option value="">请选择</option>{history.map(item => <option key={item.id} value={item.id}>{item.title} · {taskStatus(item.status)}</option>)}</select><button disabled={!linkId || busy} onClick={() => void link()}>关联到“{task.title}”</button></div> : <p className="tw-muted">未列出可关联记录。无明确所有权的旧记录需要管理员核实，不能自动认领。</p>}
    </section>}
    </div>
    <button type="button" className={`tw-to-bottom${chatScroll.away ? ' is-visible' : ''}`} aria-label="回到底部" tabIndex={chatScroll.away ? 0 : -1} aria-hidden={chatScroll.away ? undefined : true} onMouseDown={event => event.preventDefault()} onClick={chatScroll.toBottom}><ArrowDown size={16} aria-hidden="true" />{chatScroll.dot && <span className="tw-to-bottom-dot" aria-hidden="true" />}</button></div>
    {view === 'conversation' && <div className="tw-composer" ref={composer}>
      {(run?.queued_messages ?? []).filter(item => item.status === 'pending' || !item.status).map(item => (
        <div key={item.idempotency_key || item.text} className="tw-queue-preview" data-testid="tw-queue-preview" role="status">
          <span className="tw-queue-text">排队 · {queuePreview(item.text)}</span>
          <button type="button" className="tw-queue-cancel" data-testid="tw-queue-cancel" aria-label="取消排队" disabled={busy} onClick={() => void cancelQueued(String(item.idempotency_key || ''))}>×</button>
        </div>
      ))}
      <form onSubmit={event => { event.preventDefault(); void send(); }}><IncognitoStatus on={incognito} />{intent && <DraftProjectChip onRemove={removeIntent}><ExperienceChip onRemove={removeIntent} /></DraftProjectChip>}<label className="tw-sr-only" htmlFor="task-message">输入需求或追问</label><textarea id="task-message" value={draft} maxLength={20000} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (isSendKey(event)) { event.preventDefault(); void send(); } }} placeholder="说说你要推进的事" />
      <AttachmentChips files={pendingAttachments(files, sentIds)} status={composerUpload.status} error={composerUpload.error} errorCode={composerUpload.errorCode} disabled={busy || composerUpload.busy} onRemove={id => { setError(''); void bindFiles(files.filter(file => file.id !== id).map(file => file.id)).catch(cause => setError(cause instanceof Error ? cause.message : '移除失败，请重试。')); }} />
      <div className="tw-composer-bottom"><AttachButton disabled={busy || composerUpload.busy} onFiles={selected => void composerUpload.upload(selected)} />
        <IncognitoToggle on={incognito} disabled={busy} onChange={setIncognito} />
        <div className="tw-composer-send"><ModelPicker id="task-model" returnTask={task.id} list={models} value={model} closeSignal={[composerUpload.status, composerUpload.error].filter(Boolean).join('|')} platform={model === 'platform' && run?.model ? { model: run.model } : null} disabled={busy} onChange={value => { setModel(value); window.sessionStorage.setItem(LAST_MODEL_KEY, value); }} />
          {(() => { const groups = fallbackGroups(models, model); if (!groups) return null;
            const value = fallbackModelId(models, model, fallbackPick) ?? '';
            const text = value ? `备选 · ${groups[0].choices.find(item => item.value === value)?.label}` : '备选 · 不设置';
            return <Picker id="task-fallback" label="备选模型（这个模型连不上时改用它，会提示你）" value={value} text={text} groups={groups} disabled={busy}
              onChange={next => { setFallbackPick(next); window.sessionStorage.setItem(FALLBACK_MODEL_KEY, next); }} />; })()}
          <IntensityPicker id="task-intensity" value={intensity} disabled={busy} onChange={setIntensity} />
          {run && active(run.status) && <button type="button" className="tw-text-button tw-steer-btn" data-testid="tw-steer" title="立即发送" disabled={busy || !draft.trim()} onClick={() => void postLiveMessage(LIVE_MSG_MODE.immediate)}><Zap size={16} className="tw-steer-icon" aria-hidden="true" /><span className="tw-steer-label">立即发送</span></button>}
          <button type="submit" className="tw-send-icon" aria-label="发送" disabled={busy || carrying || composerUpload.busy || !draft.trim() || !model}><ArrowUp size={16} aria-hidden="true" /></button></div>{notice === '草稿已保存' && draft === task.draft && !error ? <p className={`tw-save-status${draftSavedFaded ? ' is-faded' : ''}`} role="status">草稿已保存</p> : null}</div>
      <div className="tw-composer-extra"><button type="button" className="tw-library-toggle" aria-expanded={library} aria-controls="task-library" onClick={() => setLibrary(!library)}>{libraryLabel(files.length, generatedCount)}</button>
        {serverDraft !== null ? <DraftConflict disabled={busy} onServer={takeServerDraft} onMine={() => void keepMyDraft()} />
          : autosavePaused && <DraftPaused disabled={busy} onReload={() => void reloadDraftVersion()} />}</div></form>
      {/* Outside the form: library buttons must never submit a run. */}
      <div id="task-library" className="tw-composer-library" hidden={!library}><TaskFiles selected={files} onSelect={bindFiles} /></div>
    </div>}
    {error && <p role="alert" className="tw-error">{error}</p>}{notice && notice !== '草稿已保存' ? <p role="status" className="tw-muted">{notice}</p> : null}
  </section>;
}
