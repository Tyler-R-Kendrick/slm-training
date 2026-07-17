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
  gate: { pass: boolean; passed: number; total: number } | null;
  deployment: { selected: boolean; track: string };
  inflight: { jobs: number; dispatches: number; execution: boolean };
}

export const EMPTY_HERO: HeroData = {
  reference: null,
  provenance: "committed",
  gate: null,
  deployment: { selected: false, track: "" },
  inflight: { jobs: 0, dispatches: 0, execution: false },
};

export async function fetchHero(): Promise<HeroData> {
  const [overview, jobsResp, disp]: any[] = await Promise.all([
    getJSON<any>("/api/overview").catch(() => ({})),
    getJSON<any>("/api/jobs").catch(() => ({ jobs: [], execution: false })),
    getJSON<any>("/api/dispatches").catch(() => ({ jobs: [] })),
  ]);
  const p = overview.performance ?? {};
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
      gate = {
        pass: !!g.pass,
        passed: entries.filter(Boolean).length,
        total: entries.length,
      };
    }
  }

  const dep = overview.system?.deployment ?? {};
  const activeJobs = (jobsResp.jobs ?? []).filter((j: any) =>
    ["running", "queued"].includes(j.status),
  ).length;

  return {
    reference,
    provenance: p.reference_provenance ?? "committed",
    gate,
    deployment: { selected: !!dep.selected, track: dep.selected?.track ?? "" },
    inflight: {
      jobs: activeJobs,
      dispatches: (disp.jobs ?? []).length,
      execution: !!jobsResp.execution,
    },
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
