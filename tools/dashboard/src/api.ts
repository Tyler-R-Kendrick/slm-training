import { useCallback, useEffect, useRef, useState } from "react";

export async function getJSON<T = any>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

export async function postJSON<T = any>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data && data.detail) || `${res.status} ${url}`);
  return data as T;
}

export interface PollState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Fetch `url` once, then every `ms` if > 0. `null` url is a no-op (disabled). */
export function usePoll<T = any>(url: string | null, ms = 0): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!url);

  const load = useCallback(() => {
    if (!url) return;
    getJSON<T>(url)
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(String(e?.message ?? e)))
      .finally(() => setLoading(false));
  }, [url]);

  useEffect(() => {
    if (!url) return;
    setLoading(true);
    load();
    if (ms > 0) {
      const id = setInterval(load, ms);
      return () => clearInterval(id);
    }
  }, [load, ms, url]);

  return { data, error, loading, reload: load };
}

export interface JobEvent {
  kind: "log" | "status";
  line?: string;
  status?: string;
  [k: string]: any;
}

/** Subscribe to a job's SSE stream; returns accumulated log lines + status. */
export function useJobStream(jobId: string | null) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    setLines([]);
    setStatus(null);
    if (!jobId) return;
    const es = new EventSource(`/api/jobs/${jobId}/logs`);
    esRef.current = es;
    es.addEventListener("log", (e: MessageEvent) => {
      try {
        const d = JSON.parse(e.data);
        if (typeof d.line === "string") setLines((prev) => [...prev, d.line]);
      } catch {
        /* ignore malformed frame */
      }
    });
    es.addEventListener("status", (e: MessageEvent) => {
      try {
        const d = JSON.parse(e.data);
        if (d.status) setStatus(d.status);
        if (["succeeded", "failed", "cancelled"].includes(d.status)) es.close();
      } catch {
        /* ignore */
      }
    });
    es.onerror = () => es.close();
    return () => es.close();
  }, [jobId]);

  return { lines, status };
}

export const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
