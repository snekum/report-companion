// Everything a visitor adds (pasted documents, corrections, questions) is kept
// only in this browser's localStorage. Nothing is sent anywhere. Storage can
// be unavailable (private windows, blocked site data), so every access is
// wrapped and the app keeps working in memory.
import { useCallback, useEffect, useState } from "react";

const PREFIX = "rc.v1.";
const memory = new Map<string, string>();
const listeners = new Set<() => void>();

function read(key: string): string | null {
  try {
    const v = window.localStorage.getItem(PREFIX + key);
    if (v !== null) return v;
  } catch { /* storage blocked */ }
  return memory.get(key) ?? null;
}

function write(key: string, value: string | null) {
  if (value === null) memory.delete(key); else memory.set(key, value);
  try {
    if (value === null) window.localStorage.removeItem(PREFIX + key);
    else window.localStorage.setItem(PREFIX + key, value);
  } catch { /* storage blocked: memory copy still works for this visit */ }
  listeners.forEach((l) => l());
}

export function getJSON<T>(key: string, fallback: T): T {
  const v = read(key);
  if (v === null) return fallback;
  try { return JSON.parse(v) as T; } catch { return fallback; }
}

export function setJSON(key: string, value: unknown) {
  write(key, value === undefined ? null : JSON.stringify(value));
}

export function useStored<T>(key: string, fallback: T): [T, (v: T | ((old: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => getJSON(key, fallback));
  useEffect(() => {
    setValue(getJSON(key, fallback));
    const on = () => setValue(getJSON(key, fallback));
    listeners.add(on);
    return () => { listeners.delete(on); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const set = useCallback((v: T | ((old: T) => T)) => {
    const next = typeof v === "function" ? (v as (old: T) => T)(getJSON(key, fallback)) : v;
    setJSON(key, next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return [value, set];
}

/** Removes every key this app stored on this device. */
export function forgetEverything() {
  memory.clear();
  try {
    const keys: string[] = [];
    for (let i = 0; i < window.localStorage.length; i++) {
      const k = window.localStorage.key(i);
      if (k && k.startsWith(PREFIX)) keys.push(k);
    }
    keys.forEach((k) => window.localStorage.removeItem(k));
  } catch { /* nothing stored */ }
  listeners.forEach((l) => l());
}
