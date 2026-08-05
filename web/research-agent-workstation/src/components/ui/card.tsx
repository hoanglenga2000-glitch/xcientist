import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  const componentName = (props as Record<string, unknown>)["data-ui-component"] ?? "card";
  const interactive = Boolean(props.onClick || props.role === "button" || (props as Record<string, unknown>)["data-ui-action"]);

  return (
    <div
      className={cn(
        "min-w-0 max-w-full rounded-md border border-edge/90 bg-surface-raised/96 shadow-soft",
        interactive && "cursor-pointer transition-colors duration-150 hover:border-accent/50 hover:bg-surface-raised active:bg-surface-sunken",
        className
      )}
      data-ui-component={componentName}
      {...props}
    />
  );
}

export function CardHeader({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("min-w-0 max-w-full px-3.5 pt-3.5", className)} {...props} />;
}

export function CardTitle({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("text-sm font-semibold tracking-normal text-ink", className)}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("mt-1 text-xs leading-4 text-ink-muted", className)} {...props} />;
}

export function CardContent({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  const isGrid = className?.split(/\s+/).includes("grid");
  return (
    <div
      className={cn(
        "min-w-0 max-w-full p-3.5",
        isGrid && "grid-cols-[minmax(0,1fr)]",
        className
      )}
      {...props}
    />
  );
}
