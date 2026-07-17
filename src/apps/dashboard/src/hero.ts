// Shared "mission control" hero composition: one fetch path used by BOTH the
// compiled Overview page and the interpreted overview_hero query, so the two
// renderers derive the ship-status strip from identical data.
import { useEffect, useState } from "react";
import { getJSON } from "./api";

export interface HeroReference {
  run_id: string;
  role: string;
  track: string;
  evaluation_status: string;
}

export interface HeroData {
  reference: HeroReference | null;
  provenance: string;
  gate: {
    pass: boolean;
    passed: number;
    total: number;
    /** First few failing "suite:metric" gate keys — the levers to improve next. */
    failures: string[];
    failure_count: number;
  } | null;
  deployment: { selected: boolean; track: string };
  inflight: { jobs: number; dispatches: number; execution: boolean };
  /** True when the core reads failed — unavailable evidence, not a zero state. */
  unavailable: boolean;
}

export const EMPTY_HERO: HeroData = {
  reference: null,
  provenance: "committed",
  gate: null,
  deployment: { selected: false, track: "" },
  inflight: { jobs: 0, dispatches: 0, execution: false },
  unavailable: false,
};

export async function fetchHero(): Promise<HeroData> {
  // Failed reads are flagged as unavailable rather than silently rendered as
  // a valid zero state — absence of evidence ≠ unavailable evidence.
  const [overview, jobsResp, disp]: any[] = await Promise.all([
    getJSON<any>("/api/overview").catch(() => null),
    getJSON<any>("/api/jobs").catch(() => null),
    getJSON<any>("/api/dispatches").catch(() => null),
  ]);
  const unavailable = overview === null;
  const p = overview?.performance ?? {};
  const primary = (p.references ?? [])[0] ?? null;
  const reference: HeroReference | null = primary
    ? {
        run_id: primary.run_id ?? "",
        role: primary.role ?? "",
        track: primary.track ?? "",
        evaluation_status: primary.evaluation_status ?? "",
      }
    : null;

  // Ship-gate verdict for the primary reference; absent/empty gates stay null
  // so the UI says "no gate evidence" instead of implying a pass or a fail.
  let gate: HeroData["gate"] = null;
  if (reference?.run_id) {
    const g: any = await getJSON(
      `/api/checkpoints/${encodeURIComponent(reference.run_id)}/gates`,
    ).catch(() => null);
    const entries = Object.values(g?.gates ?? {});
    if (entries.length > 0) {
      const failures = Array.isArray(g.failures) ? g.failures.map(String) : [];
      gate = {
        pass: !!g.pass,
        passed: entries.filter(Boolean).length,
        total: entries.length,
        failures: failures.slice(0, 3),
        failure_count: failures.length,
      };
    }
  }

  const dep = overview?.system?.deployment ?? {};
  const active = (jobs: any[]) =>
    jobs.filter((j: any) => ["running", "queued"].includes(j.status)).length;

  return {
    reference,
    provenance: p.reference_provenance ?? "committed",
    gate,
    deployment: { selected: !!dep.selected, track: dep.selected?.track ?? "" },
    inflight: {
      jobs: active(jobsResp?.jobs ?? []),
      dispatches: active(disp?.jobs ?? []),
      execution: !!jobsResp?.execution,
    },
    unavailable,
  };
}

/** Poll fetchHero on an interval (compiled-mode analogue of the DSL Query). */
export function useHero(intervalMs = 15000): HeroData {
  const [hero, setHero] = useState<HeroData>(EMPTY_HERO);
  useEffect(() => {
    let alive = true;
    const tick = () => fetchHero().then((h) => alive && setHero(h)).catch(() => {});
    tick();
    const timer = setInterval(tick, intervalMs);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [intervalMs]);
  return hero;
}
