import Image from "next/image";
import { cn } from "@/lib/utils";

const LOCKUP_WIDTH = 1266;
const LOCKUP_HEIGHT = 358;
const SYMBOL_CROP = 358;

export function DeepEvoGlyph({
  size = 40,
  animated = true,
  className,
  label = "DeepEvo 深度进化",
}: {
  size?: number;
  animated?: boolean;
  className?: string;
  label?: string;
}) {
  const renderedWidth = (LOCKUP_WIDTH / LOCKUP_HEIGHT) * size;

  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label || undefined}
      aria-hidden={label ? undefined : true}
      data-ui-deepevo-glyph
      className={cn("relative inline-block shrink-0 overflow-hidden rounded-full bg-black", className)}
      style={{ width: size, height: size }}
    >
      <span className={cn("absolute inset-0 overflow-hidden", animated && "deepevo-orbit")}>
        <Image
          src="/brand/deepevo-lockup.png"
          alt=""
          aria-hidden="true"
          width={LOCKUP_WIDTH}
          height={LOCKUP_HEIGHT}
          className="absolute left-0 top-0 max-w-none select-none"
          style={{ width: renderedWidth, height: size }}
          priority
        />
      </span>
    </span>
  );
}

export function DeepEvoLockup({
  height = 40,
  animated = true,
  className,
}: {
  height?: number;
  animated?: boolean;
  className?: string;
}) {
  const renderedWidth = (LOCKUP_WIDTH / LOCKUP_HEIGHT) * height;
  const wordmarkWidth = ((LOCKUP_WIDTH - SYMBOL_CROP) / LOCKUP_HEIGHT) * height;

  return (
    <span
      role="img"
      aria-label="DeepEvo 深度进化"
      data-ui-deepevo-lockup
      className={cn("inline-flex shrink-0 items-center overflow-hidden bg-black", className)}
      style={{ height }}
    >
      <DeepEvoGlyph size={height} animated={animated} label="" />
      <span className="relative block h-full overflow-hidden" style={{ width: wordmarkWidth }} aria-hidden="true">
        <Image
          src="/brand/deepevo-lockup.png"
          alt=""
          width={LOCKUP_WIDTH}
          height={LOCKUP_HEIGHT}
          className="absolute top-0 max-w-none select-none"
          style={{ left: -height, width: renderedWidth, height }}
          priority
        />
      </span>
    </span>
  );
}
