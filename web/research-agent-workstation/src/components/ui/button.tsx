import * as React from "react";
import { cn } from "@/lib/utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger" | "success";
  size?: "sm" | "md" | "icon";
};

export function Button({
  className,
  variant = "secondary",
  size = "md",
  ...props
}: ButtonProps) {
  const componentName = (props as Record<string, unknown>)["data-ui-component"] ?? "button";

  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md border text-xs font-black transition shadow-[0_1px_0_rgba(15,23,42,0.03)]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 active:translate-y-px",
        variant === "primary" &&
          "border-primary bg-primary text-accent-fg hover:bg-accent-dark",
        variant === "secondary" &&
          "border-edge bg-surface-raised text-ink hover:border-accent-muted hover:bg-accent-light/40",
        variant === "ghost" &&
          "border-transparent bg-transparent text-ink-secondary shadow-none hover:bg-surface-sunken",
        variant === "danger" &&
          "border-danger bg-danger text-frame hover:bg-danger",
        variant === "success" &&
          "border-success bg-success text-frame hover:bg-success/85",
        size === "sm" && "h-7 px-2.5",
        size === "md" && "h-8 px-3",
        size === "icon" && "h-8 w-8 p-0",
        "disabled:cursor-not-allowed disabled:opacity-55 aria-disabled:cursor-not-allowed aria-disabled:opacity-55",
        className
      )}
      data-ui-component={componentName}
      {...props}
    />
  );
}
