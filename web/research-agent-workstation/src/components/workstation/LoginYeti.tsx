"use client";

import { useEffect, useRef } from "react";

/**
 * Proto-1 login mascot (`/workspace/login-mocks/proto-1/index.html`). Decorative only: pupils follow the
 * pointer (or the caret of the focused text field), both hands cover the eyes while the password is hidden,
 * the right hand drops to "peek" when it is shown. `prefers-reduced-motion` keeps the static resting pose.
 */
export type YetiStatus = "idle" | "error" | "success";
export type YetiMotion = { cover: number; peek: number; shy: number; tilt: number; px: number; py: number };
export type YetiInput = { covering: boolean; peeking: boolean; dx: number; dy: number };

export const RESTING_YETI: YetiMotion = { cover: 0, peek: 0, shy: 0, tilt: 0, px: 0, py: 0 };
const MOUTH = { idle: "M88 138 Q100 148 112 138", covered: "M90 140 Q100 136 110 140", success: "M84 136 Q100 154 116 136" };

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const lerp = (from: number, to: number, t: number) => from + (to - from) * t;
const fmt = (value: number) => Number(value.toFixed(2));

/** One animation frame of the proto rAF loop; `dx`/`dy` are the look target relative to the face centre. */
export function stepYeti(motion: YetiMotion, { covering, peeking, dx, dy }: YetiInput): YetiMotion {
  const cover = lerp(motion.cover, covering ? 1 : 0, 0.18);
  const peek = lerp(motion.peek, peeking ? 1 : 0, 0.18);
  const shy = lerp(motion.shy, covering || peeking ? 1 : 0, 0.12);
  const tilt = lerp(motion.tilt, clamp(dx / 40, -10, 10) * (1 - shy * 0.85), 0.2);
  const reach = Math.min(Math.hypot(dx, dy) / 18, 6.5 * (1 - cover * 0.9));
  const angle = Math.atan2(dy, dx);
  const px = lerp(motion.px, Math.cos(angle) * reach * (1 - cover), 0.28);
  const py = lerp(motion.py, Math.sin(angle) * reach * (1 - cover), 0.28);
  return { cover, peek, shy, tilt, px, py };
}

export function pupilOffset(motion: YetiMotion): [number, number] {
  return [motion.px, motion.py];
}

/** Hands rest below the view and slide up over the eyes; while peeking the right hand stays lower. */
export function handTransforms({ cover, peek }: YetiMotion) {
  const raise = Math.max(cover, peek);
  const [hxL, hxR, rotL, rotR] = [14 * raise, -14 * raise, -14 * raise, 14 * raise];
  if (peek > 0.05 && cover < 0.25) {
    return {
      left: `translate(${fmt(hxL)}, ${fmt(lerp(0, -130, peek))}) rotate(${fmt(rotL)} 55 100)`,
      right: `translate(${fmt(hxR * 0.3)}, ${fmt(lerp(0, -90, peek))}) rotate(${fmt(rotR * 0.4)} 145 140)`,
    };
  }
  const dy = fmt(lerp(0, -130, cover));
  return {
    left: `translate(${fmt(hxL)}, ${dy}) rotate(${fmt(rotL)} 55 100)`,
    right: `translate(${fmt(hxR)}, ${dy}) rotate(${fmt(rotR)} 145 100)`,
  };
}

export function mouthPath(motion: YetiMotion, status: YetiStatus = "idle") {
  if (motion.cover > 0.5) return MOUTH.covered;
  return status === "success" ? MOUTH.success : MOUTH.idle;
}

function reducedMotion() {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Proto caret estimate: look at the end of the typed text in the focused (non-password) field. */
function caretPoint(fieldId: string | null | undefined) {
  const field = fieldId ? document.getElementById(fieldId) : null;
  if (!(field instanceof HTMLInputElement) || field.type === "password") return null;
  const rect = field.getBoundingClientRect();
  return { x: rect.left + 16 + Math.min((field.value.length + 1) * 8.2, rect.width - 28), y: rect.top + rect.height / 2 };
}

type LoginYetiProps = {
  passwordFocused: boolean;
  passwordVisible: boolean;
  status?: YetiStatus;
  /** Bumps on every error/success so a repeated status still shakes/bounces. */
  statusKey?: number;
  /** Id of the focused 账号 / 验证码 input; the Yeti looks at its caret instead of the pointer. */
  lookAtFieldId?: string | null;
};

export function LoginYeti({ passwordFocused, passwordVisible, status = "idle", statusKey = 0, lookAtFieldId = null }: LoginYetiProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const motionRef = useRef(RESTING_YETI);
  const live = useRef({ passwordFocused, passwordVisible, status, lookAtFieldId });
  useEffect(() => {
    live.current = { passwordFocused, passwordVisible, status, lookAtFieldId };
  });

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || reducedMotion()) return;
    const find = (selector: string) => [...svg.querySelectorAll(selector)];
    const pupils = find("[data-yeti-pupil]");
    const shines = find("[data-yeti-shine]");
    const cheeks = find("[data-yeti-cheek]");
    const lids = find("[data-yeti-lid]");
    const [face] = find("[data-yeti-face]");
    const [mouth] = find("[data-yeti-mouth]");
    const [handL] = find('[data-yeti-hand="left"]');
    const [handR] = find('[data-yeti-hand="right"]');
    let pointer = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    const onMove = (event: MouseEvent) => { pointer = { x: event.clientX, y: event.clientY }; };
    window.addEventListener("mousemove", onMove, { passive: true });

    let blink = 0;
    let frame = 0;
    let blinkTimer: ReturnType<typeof setTimeout> | undefined;
    let openTimer: ReturnType<typeof setTimeout> | undefined;
    const scheduleBlink = (delay: number) => {
      blinkTimer = setTimeout(() => {
        blink = 1;
        openTimer = setTimeout(() => { blink = 0; }, 120);
        scheduleBlink(2800 + Math.random() * 3500);
      }, delay);
    };

    const tick = () => {
      const state = live.current;
      const target = caretPoint(state.lookAtFieldId) ?? pointer;
      const box = svg.getBoundingClientRect();
      const motion = stepYeti(motionRef.current, {
        covering: state.passwordFocused && !state.passwordVisible,
        peeking: state.passwordFocused && state.passwordVisible,
        dx: target.x - (box.left + box.width / 2),
        dy: target.y - (box.top + box.height * 0.42),
      });
      motionRef.current = motion;
      svg.style.transform = `rotate(${fmt(motion.tilt)}deg)`;
      pupils.forEach((pupil, index) => {
        pupil.setAttribute("cx", String(fmt((index ? 128 : 72) + motion.px)));
        pupil.setAttribute("cy", String(fmt(102 + motion.py)));
        shines[index]?.setAttribute("cx", String(fmt((index ? 125 : 69) + motion.px * 0.7)));
        shines[index]?.setAttribute("cy", String(fmt(99 + motion.py * 0.7)));
      });
      face?.setAttribute("transform", `translate(${fmt(motion.px * 0.35 * (1 - motion.shy))}, ${fmt(motion.py * 0.25 * (1 - motion.shy))})`);
      const hands = handTransforms(motion);
      handL?.setAttribute("transform", hands.left);
      handR?.setAttribute("transform", hands.right);
      cheeks.forEach((cheek) => cheek.setAttribute("opacity", String(fmt(0.45 + 0.35 * motion.shy))));
      mouth?.setAttribute("d", mouthPath(motion, state.status));
      lids.forEach((lid) => lid.setAttribute("opacity", String(blink)));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    scheduleBlink(1800);

    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(blinkTimer);
      clearTimeout(openTimer);
      window.removeEventListener("mousemove", onMove);
      svg.style.transform = "";
    };
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || status === "idle" || typeof svg.animate !== "function" || reducedMotion()) return;
    const tilt = `rotate(${fmt(motionRef.current.tilt)}deg)`;
    if (status === "error") {
      svg.animate(["0", "-10px", "10px", "0"].map((x) => ({ transform: `${tilt} translateX(${x})` })), { duration: 420 });
    } else {
      svg.animate(["0", "-18px", "0", "-8px", "0"].map((y) => ({ transform: `rotate(0deg) translateY(${y})` })), { duration: 700 });
    }
  }, [status, statusKey]);

  return (
    <div className="relative z-[4] mx-auto h-[200px] w-[240px] overflow-visible">
      <svg
        ref={svgRef}
        viewBox="0 0 200 220"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        focusable="false"
        className="absolute bottom-0 left-1/2 -ml-[100px] h-[200px] w-[200px] origin-bottom overflow-visible"
      >
        <defs>
          <radialGradient id="login-yeti-body" cx="40%" cy="35%" r="70%"><stop offset="0%" stopColor="rgb(var(--color-success))" /><stop offset="100%" stopColor="rgb(var(--color-success-text))" /></radialGradient>
          <radialGradient id="login-yeti-ear" cx="40%" cy="40%" r="60%"><stop offset="0%" stopColor="rgb(var(--color-success))" /><stop offset="100%" stopColor="rgb(var(--color-success-text))" /></radialGradient>
        </defs>
        <ellipse cx="42" cy="78" rx="22" ry="26" fill="url(#login-yeti-ear)" />
        <ellipse cx="158" cy="78" rx="22" ry="26" fill="url(#login-yeti-ear)" />
        <ellipse cx="42" cy="78" rx="11" ry="14" fill="rgb(var(--color-success-light))" />
        <ellipse cx="158" cy="78" rx="11" ry="14" fill="rgb(var(--color-success-light))" />
        <ellipse cx="100" cy="118" rx="78" ry="72" fill="url(#login-yeti-body)" />
        <ellipse data-yeti-cheek="" cx="58" cy="128" rx="14" ry="9" fill="rgb(var(--color-danger))" opacity=".45" />
        <ellipse data-yeti-cheek="" cx="142" cy="128" rx="14" ry="9" fill="rgb(var(--color-danger))" opacity=".45" />
        <g data-yeti-face="" transform="translate(0, 0)">
          {[72, 128].map((x) => (
            <g key={x}>
              <ellipse cx={x} cy="100" rx="16" ry="18" fill="rgb(var(--color-paper))" />
              <circle data-yeti-pupil="" cx={x} cy="102" r="7" fill="rgb(var(--color-frame))" />
              <circle data-yeti-shine="" cx={x - 3} cy="99" r="2.2" fill="rgb(var(--color-paper))" />
            </g>
          ))}
          <path d="M54 82 Q72 76 88 84" fill="none" stroke="rgb(var(--color-frame-light))" strokeWidth="3.5" strokeLinecap="round" />
          <path d="M112 84 Q128 76 146 82" fill="none" stroke="rgb(var(--color-frame-light))" strokeWidth="3.5" strokeLinecap="round" />
          <path data-yeti-mouth="" d={MOUTH.idle} fill="none" stroke="rgb(var(--color-frame-light))" strokeWidth="3.2" strokeLinecap="round" />
          <ellipse cx="100" cy="118" rx="7" ry="5" fill="rgb(var(--color-frame-light))" opacity=".55" />
        </g>
        <ellipse data-yeti-lid="" cx="72" cy="100" rx="16" ry="18" fill="url(#login-yeti-body)" opacity="0" />
        <ellipse data-yeti-lid="" cx="128" cy="100" rx="16" ry="18" fill="url(#login-yeti-body)" opacity="0" />
        {([["left", 55], ["right", 145]] as const).map(([side, x]) => (
          <g key={side} data-yeti-hand={side} transform="translate(0, 0)">
            <ellipse cx={x} cy="230" rx="32" ry="26" fill="rgb(var(--color-success-light))" stroke="rgb(var(--color-success-text))" strokeWidth="2" />
            {[-15, 0, 15].map((offset) => (
              <ellipse key={offset} cx={x + offset} cy={offset ? 222 : 218} rx="9" ry="12" fill="rgb(var(--color-success-light))" stroke="rgb(var(--color-success-text))" strokeWidth="1.5" />
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}
