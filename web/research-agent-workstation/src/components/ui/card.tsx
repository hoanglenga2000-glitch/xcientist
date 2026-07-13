import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  const componentName = (props as Record<string, unknown>)["data-ui-component"] ?? "card";

  return (
    <div
      className={cn(
        "cursor-pointer rounded-md border border-slate-200/95 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.035)] transition hover:border-blue-200 hover:shadow-[0_1px_2px_rgba(15,23,42,0.05),0_10px_28px_-24px_rgba(37,99,235,0.45)] active:translate-y-px",
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
  return <div className={cn("px-3.5 pt-3.5", className)} {...props} />;
}

export function CardTitle({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("text-sm font-black tracking-normal text-slate-950", className)}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("mt-1 text-xs leading-4 text-slate-500", className)} {...props} />;
}

export function CardContent({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-3.5", className)} {...props} />;
}
