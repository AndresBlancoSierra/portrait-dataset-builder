import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface ViewerItem {
  key: string;
  src: string;
  title?: string;
  subtitle?: string;
  deleteFn?: () => void | Promise<void>;
}

interface ViewerProps {
  items: ViewerItem[];
  onExit: () => void;
  defaultTimer?: number;
}

function shuffleArray<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function Viewer({
  items,
  onExit,
  defaultTimer = 180,
}: ViewerProps) {
  const [shuffled, setShuffled] = useState(() => shuffleArray(items));
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const [timerDuration, setTimerDuration] = useState(defaultTimer);
  const [timeLeft, setTimeLeft] = useState(defaultTimer);
  const [timerDraft, setTimerDraft] = useState(String(defaultTimer));
  const [editing, setEditing] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const total = shuffled.length;
  const current = shuffled[index];
  const timerProgress = (timeLeft / timerDuration) * 100;
  const deleteFnRef = useRef(current?.deleteFn);
  deleteFnRef.current = current?.deleteFn;
  const editingRef = useRef(editing);
  editingRef.current = editing;
  const indexRef = useRef(index);
  indexRef.current = index;

  useEffect(() => {
    setShuffled(shuffleArray(items));
    setIndex(0);
    setTimeLeft(defaultTimer);
    setTimerDuration(defaultTimer);
    setTimerDraft(String(defaultTimer));
  }, [items, defaultTimer]);

  const advance = useCallback(() => {
    setIndex((i) => {
      if (i >= shuffled.length - 1) {
        setShuffled(shuffleArray(itemsRef.current));
        return 0;
      }
      return i + 1;
    });
    setTimeLeft(timerDuration);
  }, [shuffled.length, timerDuration]);

  const goBack = useCallback(() => {
    setIndex((i) => Math.max(0, i - 1));
    setTimeLeft(timerDuration);
  }, [timerDuration]);

  const removeCurrent = useCallback(() => {
    const fn = deleteFnRef.current;
    if (!fn) return;
    fn();
    const i = indexRef.current;
    setShuffled((prev) => {
      if (prev.length <= 1) {
        onExit();
        return prev;
      }
      return prev.filter((_, idx) => idx !== i);
    });
    setIndex((prev) => Math.min(prev, shuffled.length - 2));
  }, [onExit, shuffled.length]);

  useEffect(() => {
    if (paused || total === 0) return;
    timerRef.current = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          advance();
          return timerDuration;
        }
        return t - 1;
      });
    }, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [paused, advance, timerDuration, total]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onExit();
      if (e.key === ' ' || e.key === 'Enter') {
        if (editingRef.current) return;
        e.preventDefault();
        setPaused((p) => !p);
      }
      if (e.key === 'ArrowRight' || e.key === 'n') { e.preventDefault(); advance(); }
      if (e.key === 'ArrowLeft' || e.key === 'p') { e.preventDefault(); goBack(); }
      if ((e.key === 'Delete' || e.key === 'Del') && !e.repeat && deleteFnRef.current) removeCurrent();
      if (e.key === 'r') { setShuffled(shuffleArray(itemsRef.current)); setIndex(0); setTimeLeft(timerDuration); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onExit, advance, goBack, removeCurrent, timerDuration]);

  const commitTimer = useCallback((val: string) => {
    const n = parseInt(val);
    if (!isNaN(n) && n > 0) {
      setTimerDuration(n);
      setTimeLeft(n);
      setTimerDraft(String(n));
    } else {
      setTimerDraft(String(timerDuration));
    }
  }, [timerDuration]);

  if (total === 0) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black flex items-center justify-center"
      onClick={onExit}
    >
      <AnimatePresence mode="wait">
        {current && (
          <motion.img
            key={current.key}
            src={current.src}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            onClick={(e) => e.stopPropagation()}
            className="max-h-full max-w-full object-contain"
          />
        )}
      </AnimatePresence>

      {current?.title && (
        <div
          className="fixed bottom-20 right-6 text-xs text-white/40 select-none pointer-events-none"
          style={{ fontWeight: 500, letterSpacing: '0.02em' }}
        >
          {current.title}
          {current?.subtitle && <> — {current.subtitle}</>}
        </div>
      )}

      {editing ? (
        <div className="fixed top-1 right-1 z-20" onClick={(e) => e.stopPropagation()}>
          <input
            ref={inputRef}
            type="text"
            value={timerDraft}
            onChange={(e) => setTimerDraft(e.target.value)}
            onBlur={() => { commitTimer(timerDraft); setEditing(false); }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { commitTimer(timerDraft); setEditing(false); }
              if (e.key === 'Escape') { setTimerDraft(String(timerDuration)); setEditing(false); }
            }}
            className="w-12 bg-black/60 text-white/80 text-center rounded py-1 text-xs outline-none focus:ring-1 focus:ring-white/30 tabular-nums"
            autoFocus
          />
        </div>
      ) : (
        <button
          onClick={(e) => { e.stopPropagation(); setEditing(true); setTimeout(() => inputRef.current?.select(), 0); }}
          className="fixed top-1 right-1 z-20 px-2 py-1 rounded bg-black/40 text-white/50 hover:text-white/80 text-xs tabular-nums"
          title="Click to edit timer (seconds)"
        >
          {paused ? '⏸ ' : ''}{Math.ceil(timeLeft)}s
        </button>
      )}

      <div className="fixed top-0 left-0 right-0 h-0.5 bg-white/10 z-10">
        <div
          className="h-full bg-white/60 transition-all duration-1000"
          style={{ width: `${timerProgress}%` }}
        />
      </div>
    </div>
  );
}
