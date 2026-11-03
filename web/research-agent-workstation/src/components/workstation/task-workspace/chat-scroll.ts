import { useCallback, useEffect, useRef, useState } from 'react';

/** C5a L9: the message list is its own scroller; the button appears past 200px and fades out at ≤40px. */
export const BACK_TO_BOTTOM_SHOW = 200;
export const BACK_TO_BOTTOM_HIDE = 40;

export function distanceFromBottom(el: { scrollHeight: number; scrollTop: number; clientHeight: number }) {
  return el.scrollHeight - el.scrollTop - el.clientHeight;
}

/** Hysteresis: shown above 200px, hidden at ≤40px, unchanged in between. */
export function nextAway(previous: boolean, distance: number) {
  if (distance > BACK_TO_BOTTOM_SHOW) return true;
  if (distance <= BACK_TO_BOTTOM_HIDE) return false;
  return previous;
}

/** Only a reader who is already at the bottom is followed; anyone who scrolled up is never pulled back. */
export function shouldFollow(distance: number) {
  return distance <= BACK_TO_BOTTOM_HIDE;
}

const reduceMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/**
 * t238 C7c-FIX5 H3: bring a card that waits on the reader (training state, confirmation) to the top of the chat, below the
 * sticky 停止/暂停 bar (the card's scroll-margin-top leaves room for it). A reader pinned to the bottom is not followed past it afterwards.
 */
export function revealBelowBar(node: { scrollIntoView?: (options?: ScrollIntoViewOptions) => void } | null) {
  node?.scrollIntoView?.({ block: 'start', behavior: reduceMotion() ? 'auto' : 'smooth' });
}

export function useChatScroll({ enabled, streaming, resetKey }: { enabled: boolean; streaming: boolean; resetKey: unknown }) {
  const ref = useRef<HTMLDivElement>(null);
  // Distance at the user's last scroll, not after growth: new streamed content must not unpin a reader at the bottom.
  const pinned = useRef(true);
  const [away, setAway] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const onScroll = () => {
      const distance = distanceFromBottom(node);
      pinned.current = shouldFollow(distance);
      setAway(previous => nextAway(previous, distance));
    };
    const onGrow = () => {
      if (pinned.current) node.scrollTop = node.scrollHeight;
      else setAway(previous => nextAway(previous, distanceFromBottom(node)));
    };
    node.addEventListener('scroll', onScroll, { passive: true });
    // Content can also grow without a DOM mutation (fonts, images, the composer resizing the scroller): follow those too.
    const resizer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(onGrow);
    const watch = () => { resizer?.observe(node); Array.from(node.children).forEach(child => resizer?.observe(child)); };
    const observer = new MutationObserver(() => { watch(); onGrow(); });
    observer.observe(node, { childList: true, subtree: true, characterData: true });
    watch();
    return () => { node.removeEventListener('scroll', onScroll); observer.disconnect(); resizer?.disconnect(); };
  }, []);

  // A new view / run starts at the bottom of a conversation and at the top of every other tab.
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    pinned.current = enabled;
    node.scrollTop = enabled ? node.scrollHeight : 0;
    setAway(false);
  }, [resetKey, enabled]);

  const toBottom = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    pinned.current = true;
    node.scrollTo({ top: node.scrollHeight, behavior: reduceMotion() ? 'auto' : 'smooth' });
  }, []);

  return { ref, away: enabled && away, dot: enabled && away && streaming, toBottom };
}
