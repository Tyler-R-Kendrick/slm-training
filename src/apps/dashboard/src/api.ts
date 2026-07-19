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

export interface OtelEvent {
  seq?: number;
  ts?: number;
  signal?: string;
  body?: string;
  severity?: string;
  attrs?: Record<string, any>;
  [k: string]: any;
}

export interface OtelStreamState {
  events: OtelEvent[];
  status: Record<string, any> | null;
  dropped: number;
  live: boolean;
}

const OTEL_TERMINAL = new Set(["completed", "failed"]);
const OTEL_EVENT_CAP = 500;

/**
 * Subscribe to a run's OTEL SSE stream. Lazy by construction: the EventSource
 * only exists while the consuming component is mounted AND the tab is visible;
 * hiding the tab closes it and re-focusing resumes from the last seen seq.
 * A hub_epoch change (hub restarted) resets the accumulated feed.
 */
export function useOtelStream(runId: string | null): OtelStreamState {
  const [events, setEvents] = useState<OtelEvent[]>([]);
  const [status, setStatus] = useState<Record<string, any> | null>(null);
  const [dropped, setDropped] = useState(0);
  const [live, setLive] = useState(false);
  const lastSeq = useRef(0);
  const epoch = useRef<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    setEvents([]);
    setStatus(null);
    setDropped(0);
    setLive(false);
    lastSeq.current = 0;
    epoch.current = null;
    if (!runId) return;

    const checkEpoch = (d: any) => {
      if (d?.hub_epoch && epoch.current && d.hub_epoch !== epoch.current) {
        setEvents([]);
        lastSeq.current = 0;
      }
      if (d?.hub_epoch) epoch.current = d.hub_epoch;
    };

    const close = () => {
      esRef.current?.close();
      esRef.current = null;
      setLive(false);
    };

    const open = () => {
      if (esRef.current) return;
      const es = new EventSource(
        `/api/otel/runs/${encodeURIComponent(runId)}/stream?since=${lastSeq.current}`,
      );
      esRef.current = es;
      setLive(true);
      es.addEventListener("status", (e: MessageEvent) => {
        try {
          const d = JSON.parse(e.data);
          checkEpoch(d);
          setStatus(d);
          if (OTEL_TERMINAL.has(d.status)) close();
        } catch {
          /* ignore malformed frame */
        }
      });
      es.addEventListener("otel", (e: MessageEvent) => {
        try {
          const d = JSON.parse(e.data);
          checkEpoch(d);
          if (typeof d.seq === "number") {
            lastSeq.current = Math.max(lastSeq.current, d.seq);
          }
          setEvents((prev) => [...prev.slice(-(OTEL_EVENT_CAP - 1)), d]);
        } catch {
          /* ignore */
        }
      });
      es.addEventListener("dropped", (e: MessageEvent) => {
        try {
          setDropped((n) => n + (JSON.parse(e.data).count ?? 0));
        } catch {
          /* ignore */
        }
      });
      // A server-sent `error` frame carries data (terminal: close for good).
      // The built-in transport error event has none: EventSource reconnects
      // on its own unless it gave up (readyState CLOSED — e.g. a 503 or a
      // non-SSE response), where we must close or `live` sticks true.
      es.addEventListener("error", (e: any) => {
        if (e?.data || es.readyState === EventSource.CLOSED) close();
      });
    };

    const onVisibility = () => {
      if (document.hidden) close();
      else open();
    };

    open();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      close();
    };
  }, [runId]);

  return { events, status, dropped, live };
}
