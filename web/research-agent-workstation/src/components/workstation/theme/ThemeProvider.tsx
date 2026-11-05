"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as api from "@/lib/api/client";

export type ThemeMode = "dark" | "light" | "system";
export type ResolvedTheme = "dark" | "light";
type SettingsPayload = Record<string, Record<string, unknown>>;

export const THEME_STORAGE_KEY = "evomind.theme";
const STORAGE_KEY = THEME_STORAGE_KEY;

/** Inline <head> boot: sets data-theme before first paint (no flash). Default = system. */
export const THEME_BOOT_SCRIPT = `!function(){try{var k=${JSON.stringify(THEME_STORAGE_KEY)},m=localStorage.getItem(k);if(m!=="light"&&m!=="dark"&&m!=="system")m="system";var d=window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light",r=m==="system"?d:m,e=document.documentElement;e.dataset.theme=r;e.dataset.themeMode=m;e.classList.toggle("dark",r==="dark");e.style.colorScheme=r}catch(t){}}();`;

export const THEME_OPTIONS: Array<{ value: ThemeMode; label: string }> = [
  { value: "system", label: "跟随系统" },
  { value: "light", label: "浅色" },
  { value: "dark", label: "深色" },
];

function isThemeMode(value: unknown): value is ThemeMode {
  return value === "dark" || value === "light" || value === "system";
}

function resolveTheme(mode: ThemeMode): ResolvedTheme {
  if (mode !== "system") return mode;
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(mode: ThemeMode): ResolvedTheme {
  const resolved = resolveTheme(mode);
  const root = document.documentElement;
  root.dataset.theme = resolved;
  root.dataset.themeMode = mode;
  root.classList.toggle("dark", resolved === "dark");
  root.style.colorScheme = resolved;
  return resolved;
}

type ThemeContextValue = {
  mode: ThemeMode;
  persistedMode: ThemeMode;
  resolvedTheme: ResolvedTheme;
  ready: boolean;
  saving: boolean;
  previewMode: (mode: ThemeMode) => void;
  chooseMode: (mode: ThemeMode) => void;
  saveMode: (additionalSettings?: SettingsPayload) => Promise<void>;
  resetMode: () => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>("system");
  const [persistedMode, setPersistedMode] = useState<ThemeMode>("system");
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>("light");
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const generalSettingsRef = useRef<Record<string, unknown>>({});
  const previewRevisionRef = useRef(0);
  const modeRef = useRef<ThemeMode>("system");
  modeRef.current = mode;

  const previewMode = useCallback((nextMode: ThemeMode) => {
    previewRevisionRef.current += 1;
    setMode(nextMode);
    setResolvedTheme(applyTheme(nextMode));
  }, []);

  /** L10: localStorage immediate; best-effort sync to user settings if present. */
  const chooseMode = useCallback((nextMode: ThemeMode) => {
    previewRevisionRef.current += 1;
    setMode(nextMode);
    setPersistedMode(nextMode);
    window.localStorage.setItem(STORAGE_KEY, nextMode);
    setResolvedTheme(applyTheme(nextMode));
    const nextGeneral = { ...generalSettingsRef.current, theme: nextMode };
    generalSettingsRef.current = nextGeneral;
    void api.saveSettings({ general: nextGeneral }).then((payload) => {
      generalSettingsRef.current = payload.settings?.general ?? nextGeneral;
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    const revisionAtLoadStart = previewRevisionRef.current;
    const localMode = window.localStorage.getItem(STORAGE_KEY);
    const initialMode = isThemeMode(localMode) ? localMode : "system";
    setMode(initialMode);
    setPersistedMode(initialMode);
    setResolvedTheme(applyTheme(initialMode));

    api.getSettings()
      .then((payload) => {
        const general = payload.settings?.general ?? {};
        generalSettingsRef.current = general;
        const serverMode = general.theme;
        if (isThemeMode(serverMode)) {
          setPersistedMode(serverMode);
          window.localStorage.setItem(STORAGE_KEY, serverMode);
          if (previewRevisionRef.current === revisionAtLoadStart) {
            setMode(serverMode);
            setResolvedTheme(applyTheme(serverMode));
          }
        }
      })
      .catch(() => undefined)
      .finally(() => setReady(true));
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if (modeRef.current === "system") setResolvedTheme(applyTheme("system"));
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const saveMode = useCallback(async (additionalSettings: SettingsPayload = {}) => {
    setSaving(true);
    try {
      const nextGeneral = {
        ...generalSettingsRef.current,
        ...additionalSettings.general,
        theme: mode,
      };
      const payload = await api.saveSettings({
        ...additionalSettings,
        general: nextGeneral,
      });
      generalSettingsRef.current = payload.settings?.general ?? nextGeneral;
      setPersistedMode(mode);
      window.localStorage.setItem(STORAGE_KEY, mode);
      setResolvedTheme(applyTheme(mode));
    } finally {
      setSaving(false);
    }
  }, [mode]);

  const resetMode = useCallback(() => {
    setMode(persistedMode);
    setResolvedTheme(applyTheme(persistedMode));
  }, [persistedMode]);

  const value = useMemo<ThemeContextValue>(() => ({
    mode,
    persistedMode,
    resolvedTheme,
    ready,
    saving,
    previewMode,
    chooseMode,
    saveMode,
    resetMode,
  }), [mode, persistedMode, previewMode, chooseMode, ready, resetMode, resolvedTheme, saveMode, saving]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used inside ThemeProvider");
  return context;
}

/** Safe for surfaces that may render outside the provider in unit tests. */
export function useOptionalTheme() {
  return useContext(ThemeContext);
}
