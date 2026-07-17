import React, { useEffect, useMemo, useState } from "react";
import { postJSON, useJobStream } from "./api";
import type { HeroData } from "./hero";

// --- formatting helpers ----------------------------------------------------
export function fmt(v: any, digits = 3): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(digits);
  }
  return String(v);
}

export function pct(v: any): string {
  if (typeof v !== "number") return "—";
  return `${(v * 100).toFixed(0)}%`;
}

// --- layout ----------------------------------------------------------------
export function Card({
  title,
  right,
  children,
  className = "",
}: {
  title?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="card-head">
          {title && <h2 className="card-title">{title}</h2>}
          {right && <div className="card-right">{right}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Grid({ children, min = "220px" }: { children: React.ReactNode; min?: string }) {
  return (
    <div className="grid" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${min}, 1fr))` }}>
      {children}
    </div>
  );
}

export function Empty({
  children,
  ctaLabel,
  onCta,
}: {
  children: React.ReactNode;
  ctaLabel?: string;
  onCta?: () => void;
}) {
  return (
    <div className="empty">
      {children}
      {ctaLabel && onCta && (
        <div style={{ marginTop: "0.5rem" }}>
          <button className="chip" onClick={onCta}>{ctaLabel}</button>
        </div>
      )}
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="error-note">⚠ {error}</div>;
}

// --- status + provenance ---------------------------------------------------
const STATUS_CLASS: Record<string, string> = {
  running: "running",
  queued: "idle",
  cancelling: "warning",
  succeeded: "passed",
  passed: "passed",
  pass: "passed",
  true: "passed",
  failed: "failed",
  fail: "failed",
  false: "failed",
  cancelled: "idle",
  champion: "promoted",
  deployed: "promoted",
  promoted: "promoted",
  validated: "running",
  screened: "warning",
  rejected: "failed",
  collapsed: "failed",
  healthy: "passed",
  warning: "warning",
  unavailable: "idle",
};

export function StatusPill({ value, label }: { value: any; label?: string }) {
  const key = String(value).toLowerCase();
  const cls = STATUS_CLASS[key] ?? "idle";
  return <span className={`pill pill-${cls}`}>{label ?? String(value)}</span>;
}

export function ProvenanceBadge({ provenance }: { provenance?: string }) {
  if (!provenance) return null;
  const live = provenance === "live";
  return (
    <span className={`prov ${live ? "prov-live" : "prov-committed"}`} title={live ? "read from outputs/ (current)" : "committed snapshot (docs/ or src/slm_training/resources/)"}>
      {live ? "live" : "committed"}
    </span>
  );
}

// --- stat tile -------------------------------------------------------------
export function StatTile({
  label,
  value,
  sub,
  accent,
  spark,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  accent?: "moss" | "ember" | "passed" | "failed" | "running" | "promoted";
  spark?: number[];
}) {
  return (
    <div className={`tile ${accent ? `tile-${accent}` : ""}`}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {spark && spark.length > 1 && <Sparkline data={spark} />}
      {sub !== undefined && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

// --- charts (hand-rolled SVG, theme via currentColor) ----------------------
export function Sparkline({ data, width = 120, height = 28 }: { data: number[]; width?: number; height?: number }) {
  const pts = useMemo(() => {
    if (!data.length) return "";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const span = max - min || 1;
    return data
      .map((d, i) => {
        const x = (i / (data.length - 1 || 1)) * width;
        const y = height - ((d - min) / span) * (height - 4) - 2;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [data, width, height]);
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function Bars({ data }: { data: { label: string; value: number; accent?: string; help?: string }[] }) {
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <div className="bars">
      {data.map((d, i) => (
        <div className="bar-row" key={i}>
          <span className="bar-label" title={d.label}>
            {d.label}
            {d.help && <span className="bar-help" tabIndex={0} title={d.help} aria-label={`${d.label}: ${d.help}`}>ⓘ<span className="table-tooltip" role="tooltip">{d.help}</span></span>}
          </span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${(d.value / max) * 100}%`, background: d.accent }} />
          </span>
          <span className="bar-value">{fmt(d.value)}</span>
        </div>
      ))}
    </div>
  );
}

// --- dense table -----------------------------------------------------------
export function DataTable({
  columns,
  rows,
  render,
  searchable = false,
  searchPlaceholder = "Search experiments",
  maxHeight = "28rem",
}: {
  columns: { key: string; label: string; align?: "left" | "right"; help?: string; digits?: number; direction?: "higher" | "lower" }[];
  rows: any[];
  render?: Record<string, (row: any) => React.ReactNode>;
  searchable?: boolean;
  searchPlaceholder?: string;
  maxHeight?: string;
}) {
  const [query, setQuery] = React.useState("");
  if (!rows.length) return <Empty>No rows.</Empty>;
  const baseline = rows.find((row) => row.id === "P0");
  const filtered = query.trim()
    ? rows.filter((row) => Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query.toLowerCase())))
    : rows;
  return (
    <div>
      {searchable && (
        <label className="table-search">
          <span>Search experiments</span>
          <input type="search" value={query} placeholder={searchPlaceholder} onChange={(event) => setQuery(event.target.value)} />
          <span className="hint">{filtered.length} / {rows.length}</span>
        </label>
      )}
      <div className="table-wrap" style={{ maxHeight }}>
        <table className="dtable">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} style={{ textAlign: c.align ?? "left" }}>
                {c.help ? <span className="table-help" tabIndex={0} title={c.help} aria-label={`${c.label}: ${c.help}`}>{c.label}<span aria-hidden="true"> ⓘ</span><span className="table-tooltip" role="tooltip">{c.help}</span></span> : c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key} className={`cell-${c.key}`} style={{ textAlign: c.align ?? "left" }}>
                  {render && render[c.key] ? render[c.key](row) : c.direction ? <MetricCell row={row} baseline={baseline} column={c} /> : fmt(row[c.key], c.digits ?? 3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        </table>
      </div>
    </div>
  );
}

type MetricColumn = { key: string; label: string; help?: string; digits?: number; direction?: "higher" | "lower" };

function MetricCell({ row, baseline, column }: { row: any; baseline: any; column: MetricColumn }) {
  const value = row[column.key];
  const numeric = typeof value === "number" ? value : Number(value);
  let state = "neutral";
  let insight = "Within the neutral comparison band.";
  if (!Number.isFinite(numeric)) {
    state = "info";
    insight = "No numeric value was recorded for this metric.";
  } else if (!baseline || typeof baseline[column.key] !== "number") {
    state = "info";
    insight = "No numeric P0 baseline is available for this comparison.";
  } else if (row.id !== "P0") {
    const base = baseline[column.key];
    if (base === 0) {
      state = "warning";
      insight = row.guardrails?.note || "P0 is zero, so a relative performance range cannot be established.";
    } else {
      const change = (numeric - base) / Math.abs(base);
      const favorable = column.direction === "lower" ? -change : change;
      state = favorable > 0.05 ? "good" : favorable < -0.05 ? "bad" : "neutral";
      insight = `${favorable >= 0 ? "+" : ""}${(favorable * 100).toFixed(1)}% vs P0; ${column.direction === "lower" ? "lower" : "higher"} is better.`;
      if (column.key === "parse_rate" && row.guardrails?.note) {
        state = "warning";
        insight = row.guardrails.note;
      }
    }
  }
  return <span className={`metric-value metric-${state}`}>
    {fmt(value, column.digits ?? 3)}
    {state !== "neutral" && <span className="metric-flag" tabIndex={0} aria-label={insight} title={insight}>{state === "good" ? "✓" : state === "bad" ? "↓" : state === "warning" ? "⚠" : "ⓘ"}<span className="table-tooltip" role="tooltip">{insight}</span></span>}
  </span>;
}

// --- gate matrix -----------------------------------------------------------
export interface GatePayload {
  policy?: Record<string, Record<string, number>>;
  actual?: Record<string, Record<string, any>>;
  gates?: Record<string, boolean>;
  failures?: string[];
  pass?: boolean;
}

export function GateMatrix({ gate }: { gate: GatePayload }) {
  const policy = gate.policy ?? {};
  const actual = gate.actual ?? {};
  const suites = Object.keys(policy);
  const metrics = Array.from(new Set(suites.flatMap((s) => Object.keys(policy[s] ?? {}))));
  if (!suites.length) return <Empty>No gate policy.</Empty>;
  return (
    <div className="table-wrap">
      <table className="dtable gate-matrix">
        <thead>
          <tr>
            <th>suite</th>
            {metrics.map((m) => <th key={m} style={{ textAlign: "right" }}>{m}</th>)}
          </tr>
        </thead>
        <tbody>
          {suites.map((s) => (
            <tr key={s}>
              <td className="mono">{s}</td>
              {metrics.map((m) => {
                const need = policy[s]?.[m];
                if (need === undefined) return <td key={m} className="gate-na">·</td>;
                const got = actual[s]?.[m];
                const ok = gate.gates?.[`${s}:${m}`];
                return (
                  <td key={m} className={`gate-cell gate-${ok ? "pass" : "fail"}`} style={{ textAlign: "right" }} title={`need ≥ ${need}`}>
                    <span className="gate-glyph">{ok ? "✓" : "✕"}</span> {fmt(got, 2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- lifecycle timeline ----------------------------------------------------
const LIFECYCLE = ["running", "screened", "validated", "champion", "deployed"];
export function Timeline({ state }: { state?: string | null }) {
  const idx = state ? LIFECYCLE.indexOf(state) : -1;
  const rejected = state === "rejected";
  return (
    <ol className="timeline">
      {LIFECYCLE.map((s, i) => (
        <li key={s} className={`tl-step ${i <= idx ? "tl-done" : ""} ${i === idx ? "tl-current" : ""}`}>
          <span className="tl-dot" />
          <span className="tl-label">{s}</span>
        </li>
      ))}
      {rejected && <li className="tl-step tl-rejected"><span className="tl-dot" /><span className="tl-label">rejected</span></li>}
    </ol>
  );
}

// --- log stream (SSE) ------------------------------------------------------
export function LogStream({ jobId }: { jobId: string | null }) {
  const { lines, status } = useJobStream(jobId);
  const ref = React.useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines.length]);
  if (!jobId) return null;
  return (
    <div className="logstream">
      <div className="logstream-head">
        <span className="mono">job {jobId}</span>
        {status && <StatusPill value={status} />}
      </div>
      <pre ref={ref} className="logstream-body">{lines.join("\n") || "…"}</pre>
    </div>
  );
}

// --- threshold editor ------------------------------------------------------
export function ThresholdEditor({
  policy,
  onChange,
}: {
  policy: Record<string, Record<string, number>>;
  onChange: (next: Record<string, Record<string, number>>) => void;
}) {
  const suites = Object.keys(policy);
  return (
    <div className="thresholds">
      {suites.map((s) => (
        <div className="thr-suite" key={s}>
          <div className="thr-suite-name mono">{s}</div>
          <div className="thr-metrics">
            {Object.entries(policy[s]).map(([m, val]) => (
              <label className="thr-metric" key={m}>
                <span>{m}</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={val}
                  onChange={(e) => {
                    const next = JSON.parse(JSON.stringify(policy));
                    next[s][m] = Number(e.target.value);
                    onChange(next);
                  }}
                />
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// --- job launcher (allowlist-driven) --------------------------------------
export interface JobDef {
  job: string;
  kind: string;
  summary: string;
  positional: string[];
  params: Record<string, { type: string; choices?: string[]; min?: number; max?: number }>;
}

export function JobLauncher({
  jobDef,
  execution,
  onLaunched,
  defaults = {},
}: {
  jobDef: JobDef;
  execution: boolean;
  onLaunched: (jobId: string) => void;
  defaults?: Record<string, any>;
}) {
  const [params, setParams] = useState<Record<string, any>>(defaults);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const set = (k: string, v: any) => setParams((p) => ({ ...p, [k]: v }));

  async function launch() {
    setBusy(true);
    setErr(null);
    try {
      const job = await postJSON("/api/jobs", { job: jobDef.job, params });
      onLaunched(job.id);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="launcher">
      <div className="launcher-head">
        <span className="mono launcher-name">{jobDef.job}</span>
        {jobDef.kind === "dispatch" && <span className="pill pill-warning">dispatch</span>}
      </div>
      <p className="launcher-summary">{jobDef.summary}</p>
      <div className="launcher-params">
        {Object.entries(jobDef.params).map(([name, rule]) => (
          <label className="launcher-field" key={name}>
            <span>{name}</span>
            {rule.type === "Choice" ? (
              <select value={params[name] ?? ""} onChange={(e) => set(name, e.target.value)}>
                <option value="">—</option>
                {rule.choices?.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : rule.type === "Flag" || rule.type === "BooleanOptionalFlag" ? (
              <input type="checkbox" checked={!!params[name]} onChange={(e) => set(name, e.target.checked)} />
            ) : rule.type === "IntRange" ? (
              <input type="number" min={rule.min} max={rule.max} value={params[name] ?? ""} onChange={(e) => set(name, Number(e.target.value))} />
            ) : (
              <input type="text" value={params[name] ?? ""} onChange={(e) => set(name, e.target.value)} />
            )}
          </label>
        ))}
      </div>
      {err && <div className="error-note">{err}</div>}
      <button className="btn btn-primary" disabled={!execution || busy} onClick={launch} title={execution ? "" : "read-only deployment"}>
        {busy ? "launching…" : execution ? "Run" : "Read-only"}
      </button>
    </div>
  );
}

// --- mission-control hero ----------------------------------------------------
// Shared by compiled Overview and the interpreted HeroStrip component so both
// renderers produce identical markup from the same HeroData (see hero.ts).
export function HeroStrip({
  hero,
  navigate,
}: {
  hero: HeroData;
  navigate?: (to: string) => void;
}) {
  const ref = hero.reference;
  const gate = hero.gate;
  const verdictClass = gate ? (gate.pass ? "verdict-pass" : "verdict-fail") : "verdict-none";
  const verdictText = gate
    ? `GATES ${gate.pass ? "PASS" : "FAIL"} ${gate.passed}/${gate.total}`
    : "NO GATE EVIDENCE";
  const flight = hero.inflight;
  return (
    <section className={`hero hero-${gate ? (gate.pass ? "pass" : "fail") : "none"}`} aria-label="Ship status">
      <div className="hero-verdict">
        <span className={`verdict ${verdictClass}`}>{verdictText}</span>
        <span className="hero-verdict-sub">ship gates · current reference</span>
        {gate && !gate.pass && gate.failures.length > 0 && (
          <span className="hero-failures mono">
            failing: {gate.failures.join(", ")}
            {gate.failure_count > gate.failures.length ? ` +${gate.failure_count - gate.failures.length} more` : ""}
          </span>
        )}
      </div>
      <div className="hero-ref">
        {ref ? (
          <>
            <div className="hero-run">
              <a
                className="mono runlink"
                onClick={() => ref.run_id && navigate?.(`/runs/${encodeURIComponent(ref.run_id)}`)}
              >
                {ref.run_id || "—"}
              </a>
              <ProvenanceBadge provenance={hero.provenance} />
            </div>
            <div className="hero-meta">
              {[ref.role, ref.track, ref.evaluation_status].filter(Boolean).join(" · ")}
            </div>
            <div className="hero-meta">
              {hero.deployment.selected ? `deployed: ${hero.deployment.track || "selected"}` : "nothing deployed"}
            </div>
          </>
        ) : (
          <>
            <div className="hero-meta">No evaluated reference yet — run a smoke suite, then evaluate ship gates.</div>
            <button className="chip" onClick={() => navigate?.("/smoke")}>run smoke →</button>
          </>
        )}
      </div>
      <div className="hero-side">
        <div className="hero-flight">
          <span className={`pill ${flight.jobs ? "pill-running" : "pill-idle"}`}>
            {flight.jobs} live job{flight.jobs === 1 ? "" : "s"}
          </span>
          <span className={`pill ${flight.dispatches ? "pill-warning" : "pill-idle"}`}>
            {flight.dispatches} remote
          </span>
          <span className="hint">{flight.execution ? "control plane online" : "read-only"}</span>
        </div>
        <div className="hero-ctas">
          <button className="chip" onClick={() => navigate?.("/checkpoints")}>gates &amp; promotion →</button>
          <button className="chip" onClick={() => navigate?.("/experiments")}>experiments →</button>
        </div>
      </div>
    </section>
  );
}

// --- in-flight lists ---------------------------------------------------------
// Plain views shared verbatim by compiled Overview and the interpreted
// JobList / JobsBadge / DispatchList wrappers in interpret/library.tsx.
export interface JobRows {
  rows?: { id?: string | number; job?: string; status?: string }[];
  execution?: boolean;
}

export function JobLines({ data }: { data: JobRows }) {
  const rows = data.rows ?? [];
  if (!data.execution) return <Empty>Execution disabled — serve locally to run jobs.</Empty>;
  if (!rows.length) return <Empty>No jobs running.</Empty>;
  return (
    <>
      {rows.map((j, i) => (
        <div
          className="job-line"
          key={j.id ?? i}
          style={{ display: "flex", justifyContent: "space-between", padding: "0.35rem 0", borderBottom: "1px solid var(--border)" }}
        >
          <span className="mono">{j.job}</span>
          <span style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <StatusPill value={j.status} />
            {j.id != null && (
              <button className="chip" onClick={() => postJSON(`/api/jobs/${j.id}/cancel`, {})}>
                cancel
              </button>
            )}
          </span>
        </div>
      ))}
    </>
  );
}

export function JobsBadgeView({ data }: { data: JobRows }) {
  return data.execution ? (
    <StatusPill value="running" label="control plane" />
  ) : (
    <span className="prov prov-committed">read-only</span>
  );
}

export interface DispatchRows {
  rows?: { id?: string | number; job?: string; status?: string; url?: string }[];
  remotes?: { run_id?: string; url?: string }[];
}

export function DispatchLines({
  data,
  navigate,
}: {
  data: DispatchRows;
  navigate?: (to: string) => void;
}) {
  const jobs = data.rows ?? [];
  const remotes = data.remotes ?? [];
  if (!jobs.length && !remotes.length) {
    return (
      <Empty
        ctaLabel={navigate ? "launch one from Experiments →" : undefined}
        onCta={navigate ? () => navigate("/experiments") : undefined}
      >
        No remote (HF Jobs / pod) trains dispatched{navigate ? "." : " — launch one from Experiments."}
      </Empty>
    );
  }
  return (
    <>
      {jobs.map((j, i) => (
        <div key={j.id ?? `j${i}`} className="dispatch-row">
          <span className="mono">{j.job}</span>
          <span style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
            <StatusPill value={j.status} />
            {j.url && (
              <a className="runlink" href={j.url} target="_blank" rel="noreferrer">
                view remote ↗
              </a>
            )}
          </span>
        </div>
      ))}
      {remotes.map((r, i) => (
        <div key={r.run_id ?? `r${i}`} className="dispatch-row">
          <span className="mono">{r.run_id}</span>
          {r.url && (
            <a className="runlink" href={r.url} target="_blank" rel="noreferrer">
              durable checkpoint ↗
            </a>
          )}
        </div>
      ))}
    </>
  );
}
