import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/data/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        border: "#E2E8F0",
        background: "#F8FAFC",
        foreground: "#0F172A",
        muted: "#64748B",
        navy: "#071524",
        primary: "#1D4ED8",
        success: "#047857",
        warning: "#B45309",
        danger: "#B91C1C"
      },
      boxShadow: {
        soft: "0 10px 30px rgba(15, 23, 42, 0.06)"
      }
    }
  },
  plugins: []
};

export default config;
