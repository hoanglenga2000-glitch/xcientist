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

const STORAGE_KEY = "evomind.theme";

function isThemeMode(value: unknown): value is ThemeMode {
  return value === "dark" || value === "light" || value === "system";
}

function resolveTheme(mode: ThemeMode): ResolvedTheme {
  if (mode !== "system") return mode;
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
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
  saveMode: (additionalSettings?: SettingsPayload) => Promise<void>;
  resetMode: () => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>("dark");
  const [persistedMode, setPersistedMode] = useState<ThemeMode>("dark");
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>("dark");
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const generalSettingsRef = useRef<Record<string, unknown>>({});
  const previewRevisionRef = useRef(0);

  const previewMode = useCallback((nextMode: ThemeMode) => {
    previewRevisionRef.current += 1;
    setMode(nextMode);
    setResolvedTheme(applyTheme(nextMode));
  }, []);

  useEffect(() => {
    const revisionAtLoadStart = previewRevisionRef.current;
    const localMode = window.localStorage.getItem(STORAGE_KEY);
    const initialMode = isThemeMode(localMode) ? localMode : "dark";
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
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => {
      if (mode === "system") setResolvedTheme(applyTheme("system"));
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [mode]);

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
    saveMode,
    resetMode,
  }), [mode, persistedMode, previewMode, ready, resetMode, resolvedTheme, saveMode, saving]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used inside ThemeProvider");
  return context;
}
