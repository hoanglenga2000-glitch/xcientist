'use client';
import { THEME_OPTIONS, useTheme, type ThemeMode } from './ThemeProvider';

/** L10 / L11: 3-segment 跟随系统 / 浅色 / 深色. */
export function ThemeSegment({ id = "workspace-theme" }: { id?: string }) {
  const theme = useTheme();
  return (
    <div className="tw-theme-seg" role="radiogroup" aria-label="主题" id={id}>
      {THEME_OPTIONS.map(option => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={theme.mode === option.value}
          data-theme-option={option.value}
          onClick={() => theme.chooseMode(option.value as ThemeMode)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
