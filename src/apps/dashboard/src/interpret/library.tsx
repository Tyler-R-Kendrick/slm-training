// Hybrid OpenUI component library for interpreted mode.
//
// createLibrary(stock openuiLibrary components + custom wrappers of the dashboard's
// own React components) so interpreted mode renders the SAME components (dark tokens,
// same behaviour) as compiled -> pixel parity. lang-core uses the zod/v4 API.
//
// A component renderer receives { props, renderNode, statementId } (NOT props spread).
// Nested child components arrive as ElementNode descriptors and must be turned into
// React nodes via renderNode(value). Data (Query results, arrays) arrive as plain
// values on props.
import React, { useEffect, useState } from "react";
import { z } from "zod/v4";
import { createLibrary, defineComponent, useStateField, reactive } from "@openuidev/react-lang";
import { openuiLibrary } from "@openuidev/react-ui";
import {
  Card,
  Grid as GridC,
  StatTile as StatTileC,
  StatusPill as StatusPillC,
  ProvenanceBadge as ProvenanceBadgeC,
  Bars as BarsC,
  Sparkline as SparklineC,
  DataTable as DataTableC,
  GateMatrix as GateMatrixC,
  Timeline as TimelineC,
  ThresholdEditor as ThresholdEditorC,
  JobLauncher as JobLauncherC,
  LogStream as LogStreamC,
  Empty,
  fmt,
  pct,
  type GatePayload,
} from "../components";
import { getJSON, postJSON } from "../api";
import { useCaps, jobDef } from "../caps";
import { Playground } from "../pages/Playground";
import { TrainingDataBrowser, TrainingDataWorkspace } from "../pages/Data";
import { navRef } from "./nav";

const any = z.any();
const nodeList = z.array(any);

function rowsOf(v: any): any[] {
  if (Array.isArray(v)) return v;
  if (v && Array.isArray(v.rows)) return v.rows;
  return [];
}

// Render a children value (single ElementNode or array of them) to React nodes.
function kids(children: any, renderNode: (v: unknown) => React.ReactNode): React.ReactNode {
  const arr = Array.isArray(children) ? children : children == null ? [] : [children];
  return arr.map((c, i) => <React.Fragment key={i}>{renderNode(c)}</React.Fragment>);
}

// Top-level page container: a plain <div> so the app's CSS margins drive vertical
// rhythm exactly as in compiled mode (a stock Stack would double-space via its gap).
const Page = defineComponent({
  name: "Page",
  description: "Top-level page wrapper (plain div) matching each compiled page's outer container.",
  props: z.object({ children: nodeList.optional() }),
  component: ({ props, renderNode }: any) => <div>{kids(props.children, renderNode)}</div>,
});

// Page header: h1.page-title + p.page-sub, matching every compiled page's <div.page-head>.
const PageHead = defineComponent({
  name: "PageHead",
  description: "Page header block: big title + muted sub-paragraph (compiled `.page-head`).",
  props: z.object({ title: z.string(), sub: z.string().optional() }),
  component: ({ props }: any) => (
    <div className="page-head">
      <h1 className="page-title">{props.title}</h1>
      {props.sub ? <p className="page-sub">{props.sub}</p> : null}
    </div>
  ),
});

// Small provenance-style badge span (kind = live | committed), e.g. the "read-only"
// header badge on the Live-jobs card.
const Badge = defineComponent({
  name: "Badge",
  description: "Small badge span styled like a provenance badge. kind = live | committed.",
  props: z.object({ text: z.string(), kind: z.string().optional() }),
  component: ({ props }: any) => (
    <span className={`prov prov-${props.kind || "committed"}`}>{props.text}</span>
  ),
});

const Panel = defineComponent({
  name: "Panel",
  description:
    "Dashboard card/section: title, child content, and an optional right-header slot (badge/chip/link node or array).",
  props: z.object({
    title: z.string().optional(),
    children: nodeList.optional(),
    right: any.optional(),
  }),
  component: ({ props, renderNode }: any) => (
    <Card title={props.title} right={props.right != null ? kids(props.right, renderNode) : undefined}>
      {kids(props.children, renderNode)}
    </Card>
  ),
});

const Grid = defineComponent({
  name: "Grid",
  description: "Responsive auto-fill grid of children (stat tiles, cards).",
  props: z.object({ children: any.optional(), min: z.string().optional() }),
  component: ({ props, renderNode }: any) => (
    <GridC min={props.min || "200px"}>{kids(props.children, renderNode)}</GridC>
  ),
});

const Row = defineComponent({
  name: "Row",
  description: "Two-column responsive row of children.",
  props: z.object({ children: nodeList.optional() }),
  component: ({ props, renderNode }: any) => (
    <div className="two-col">{kids(props.children, renderNode)}</div>
  ),
});

const StatTile = defineComponent({
  name: "StatTile",
  description: "A metric tile: label, big value (string or number), optional sub-text and accent.",
  props: z.object({
    label: z.string(),
    value: any,
    sub: z.string().optional(),
    accent: z.string().optional(),
  }),
  // Preserve compiled's undefined (no `.tile-sub` div) vs "" (empty div) distinction.
  component: ({ props }: any) => (
    <StatTileC label={props.label} value={props.value} sub={props.sub == null ? undefined : props.sub} accent={props.accent || undefined} />
  ),
});

// Inline muted note span (optionally mono) for card-header right slots.
const Note = defineComponent({
  name: "Note",
  description: "Inline muted note span; cls defaults to 'hint' (pass 'hint mono' for run ids).",
  props: z.object({ text: z.string().optional(), cls: z.string().optional() }),
  component: ({ props }: any) => <span className={props.cls || "hint"}>{props.text ?? ""}</span>,
});

// Empty-state note (matches compiled `.empty` placeholders in card bodies).
const EmptyNote = defineComponent({
  name: "EmptyNote",
  description: "Muted empty-state placeholder for a card body.",
  props: z.object({ text: z.string() }),
  component: ({ props }: any) => <Empty>{props.text}</Empty>,
});

// Literal status pill (fixed label + class), e.g. the "remote" warning pill.
const Pill = defineComponent({
  name: "Pill",
  description: "Literal pill: text + kind (running|idle|warning|passed|failed|promoted).",
  props: z.object({ text: z.string(), kind: z.string().optional() }),
  component: ({ props }: any) => <span className={`pill pill-${props.kind || "idle"}`}>{props.text}</span>,
});

const StatusPill = defineComponent({
  name: "StatusPill",
  description: "Coloured status pill.",
  props: z.object({ value: z.string(), label: z.string().optional() }),
  component: ({ props }: any) => <StatusPillC value={props.value} label={props.label} />,
});

const ProvenanceBadge = defineComponent({
  name: "ProvenanceBadge",
  description: "live | committed provenance badge.",
  props: z.object({ provenance: z.string().optional() }),
  component: ({ props }: any) => <ProvenanceBadgeC provenance={props.provenance} />,
});

const Bars = defineComponent({
  name: "Bars",
  description: "Horizontal bar list from a data row-set of {label,value}.",
  props: z.object({ data: any }),
  component: ({ props }: any) => <BarsC data={rowsOf(props.data)} />,
});

const Sparkline = defineComponent({
  name: "Sparkline",
  description: "Inline sparkline from an array of numbers.",
  props: z.object({ data: any }),
  component: ({ props }: any) => <SparklineC data={rowsOf(props.data).map(Number)} />,
});

const Timeline = defineComponent({
  name: "Timeline",
  description: "Lifecycle stepper.",
  props: z.object({ state: z.string().optional() }),
  component: ({ props }: any) => <TimelineC state={props.state} />,
});

const DataTable = defineComponent({
  name: "DataTable",
  description:
    "Dense table. columns = [{key,label,align,help,digits,direction}]; rows = a Query row-set; statusLen truncates status-pill labels (default 26); linkKey renders that column as a mono run-link; searchable enables filtering.",
  props: z.object({ columns: z.array(any), rows: any, statusLen: z.number().optional(), linkKey: z.string().optional(), searchable: z.boolean().optional(), searchPlaceholder: z.string().optional() }),
  component: ({ props }: any) => {
    const cols = (props.columns || []).filter((c: any) => c && c.key);
    const data = rowsOf(props.rows);
    const statusLen = props.statusLen ?? 26;
    const linkKey = props.linkKey;
    if (!cols.length) return <Empty>No columns.</Empty>;
    return (
      <DataTableC
        columns={cols as any}
        rows={data}
        searchable={props.searchable}
        searchPlaceholder={props.searchPlaceholder}
        render={Object.fromEntries(
          cols.filter((c: any) => !c.direction).map((c: any) => [
            c.key,
            (r: any) => {
              if (linkKey && c.key === linkKey) {
                const to = `/runs/${encodeURIComponent(r.run_id || r[c.key])}`;
                return (
                  <a className="mono runlink" title="open run detail" onClick={() => navRef.current?.(to)}>
                    {fmt(r[c.key])}
                  </a>
                );
              }
              if (c.key === "status" || String(c.key).endsWith("_status")) {
                return <StatusPillC value={r[c.key]} label={r[c.key] ? String(r[c.key]).slice(0, statusLen) : "—"} />;
              }
              if (c.key === "description" || c.key === "prompt") {
                return <span style={{ color: "var(--text-dim)" }}>{fmt(r[c.key])}</span>;
              }
              return <span className={c.key === "id" || c.key === "run_id" ? "mono" : ""}>{fmt(r[c.key])}</span>;
            },
          ])
        )}
      />
    );
  },
});

const GateMatrix = defineComponent({
  name: "GateMatrix",
  description: "Ship-gate pass/fail matrix from an evaluate_ship_gates payload.",
  props: z.object({ gate: any }),
  component: ({ props }: any) => <GateMatrixC gate={props.gate || {}} />,
});

// Header chip / link. Internal hrefs ("/experiments") navigate in-app with the
// exact "chip" styling; external (http…) render as a "runlink" that opens a tab.
const NavChip = defineComponent({
  name: "NavChip",
  description: "Header action: internal href navigates in-app (chip); external href opens a new tab (runlink).",
  props: z.object({
    label: z.string(),
    href: z.string().optional(),
    external: z.boolean().optional(),
    top: z.boolean().optional(),
  }),
  component: ({ props }: any) => {
    const href = props.href || "";
    const external = props.external || /^https?:/i.test(href);
    if (external) {
      return (
        <a className="runlink" href={href || undefined} target="_blank" rel="noreferrer">
          {props.label}
        </a>
      );
    }
    return (
      <button
        className="chip"
        style={props.top ? { marginTop: "0.6rem" } : undefined}
        onClick={() => href && navRef.current?.(href)}
      >
        {props.label}
      </button>
    );
  },
});

const Hint = defineComponent({
  name: "Hint",
  description: "Muted helper caption (matches compiled `.hint` paragraphs).",
  props: z.object({ text: z.string(), top: z.boolean().optional() }),
  component: ({ props }: any) => (
    <p className="hint" style={props.top ? { marginTop: "0.6rem" } : undefined}>
      {props.text}
    </p>
  ),
});

// Live jobs list. data = overview_jobs Query result {rows:[{id,job,status}], execution}.
// Mirrors the compiled Live-jobs card: read-only + empty states, then job lines
// with a status pill and a working cancel chip when the control plane is online.
const JobList = defineComponent({
  name: "JobList",
  description: "Active-job lines from the overview_jobs query, with read-only / empty states and a cancel chip.",
  props: z.object({ data: any }),
  component: ({ props }: any) => {
    const d = props.data || {};
    const execution = !!d.execution;
    const rows = rowsOf(d);
    if (!execution) return <Empty>Execution disabled — serve locally to run jobs.</Empty>;
    if (!rows.length) return <Empty>No jobs running.</Empty>;
    return (
      <>
        {rows.map((j: any, i: number) => (
          <div
            className="job-line"
            key={j.id ?? i}
            style={{ display: "flex", justifyContent: "space-between", padding: "0.35rem 0", borderBottom: "1px solid var(--border)" }}
          >
            <span className="mono">{j.job}</span>
            <span style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <StatusPillC value={j.status} />
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
  },
});

// Live-jobs header badge: "control plane" pill when the control plane is online,
// else a muted "read-only" badge — mirrors the compiled card's `right` slot.
const JobsBadge = defineComponent({
  name: "JobsBadge",
  description: "Header badge for the Live-jobs card, driven by the overview_jobs query's execution flag.",
  props: z.object({ data: any }),
  component: ({ props }: any) =>
    props.data && props.data.execution ? (
      <StatusPillC value="running" label="control plane" />
    ) : (
      <span className="prov prov-committed">read-only</span>
    ),
});

// Remote dispatches list. data = dispatches Query result {rows:[{id,job,status,url}],
// remotes:[{run_id,url}]}. Mirrors the compiled Remote-dispatches card.
const DispatchList = defineComponent({
  name: "DispatchList",
  description: "Remote (HF Jobs / pod) dispatch rows + durable checkpoints from the dispatches query.",
  props: z.object({ data: any }),
  component: ({ props }: any) => {
    const d = props.data || {};
    const jobs = rowsOf(d);
    const remotes = Array.isArray(d.remotes) ? d.remotes : [];
    if (!jobs.length && !remotes.length) {
      return <Empty>No remote (HF Jobs / pod) trains dispatched — launch one from Experiments.</Empty>;
    }
    return (
      <>
        {jobs.map((j: any, i: number) => (
          <div key={j.id ?? `j${i}`} className="dispatch-row">
            <span className="mono">{j.job}</span>
            <span style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
              <StatusPillC value={j.status} />
              {j.url && (
                <a className="runlink" href={j.url} target="_blank" rel="noreferrer">
                  view remote ↗
                </a>
              )}
            </span>
          </div>
        ))}
        {remotes.map((r: any, i: number) => (
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
  },
});

// ---- Interactive / stateful widgets --------------------------------------
// These delegate to real React FCs (below) so hooks + state work, then wrap them
// with defineComponent. This is the "custom-wrap bespoke widget" half of Hybrid.

// Reactive chip-row selector two-way bound to an OpenUI $variable. The `value` prop is
// `reactive(...)`, so the DSL passes the $variable itself (e.g. ChipTabs($kind, [...])):
// clicking a chip writes the shared reactive store, and any Query({ arg: $kind }, …)
// elsewhere in the program re-fetches automatically. Same mechanism stock inputs use.
function ChipTabsImpl({ binding, options }: { binding: unknown; options: string[] }) {
  const field: any = useStateField("chipTabs", binding);
  if (!options || options.length === 0) return null;
  const current = field?.value ?? options[0];
  return (
    <div className="chip-row" style={{ marginBottom: "1rem" }}>
      {(options || []).map((o) => (
        <span key={o} className={`chip ${o === current ? "active" : ""}`} onClick={() => field?.setValue?.(o)}>
          {o}
        </span>
      ))}
    </div>
  );
}

const ChipTabs = defineComponent({
  name: "ChipTabs",
  description: "Selector chip row two-way bound to a $variable: ChipTabs($var, [options]); clicking re-runs Query args that read $var.",
  props: z.object({ value: reactive(z.string()), options: z.array(z.string()) }),
  component: ({ props }: any) => <ChipTabsImpl binding={props.value} options={props.options} />,
});

// Job launcher(s) + shared SSE log stream. jobs = [{ job, defaults }]; only jobs on
// the capabilities allowlist render, matching the compiled `jobDef(caps, name) && …`.
function JobConsoleImpl({ jobs, emptyText }: { jobs: { job: string; defaults?: Record<string, unknown> }[]; emptyText?: string }) {
  const caps = useCaps();
  const [jobId, setJobId] = useState<string | null>(null);
  const defs = (jobs || [])
    .map((j) => ({ def: jobDef(caps, j.job), defaults: j.defaults || {} }))
    .filter((x) => x.def);
  return (
    <>
      {defs.length > 0 ? (
        <div className="two-col">
          {defs.map(({ def, defaults }, i) => (
            <JobLauncherC key={i} jobDef={def!} execution={caps.execution} defaults={defaults} onLaunched={setJobId} />
          ))}
        </div>
      ) : emptyText ? (
        <Empty>{emptyText}</Empty>
      ) : null}
      <LogStreamC jobId={jobId} />
    </>
  );
}

const JobConsole = defineComponent({
  name: "JobConsole",
  description: "Allowlist-driven job launcher grid + shared SSE log stream. jobs = [{ job, defaults }]; emptyText shows when none are allowed.",
  props: z.object({ jobs: any, emptyText: z.string().optional() }),
  component: ({ props }: any) => <JobConsoleImpl jobs={props.jobs || []} emptyText={props.emptyText} />,
});

const DataBrowser = defineComponent({
  name: "DataBrowser",
  description: "Paginated, searchable training-record browser with full record inspection.",
  props: z.object({ version: z.string().optional() }),
  component: ({ props }: any) => <TrainingDataBrowser version={props.version} />,
});

const DataGenerator = defineComponent({
  name: "DataGenerator",
  description: "Simple training-dataset creation control with safe defaults.",
  props: z.object({ versions: z.array(z.string()).optional() }),
  component: ({ props }: any) => (
    <TrainingDataWorkspace versions={props.versions || []} />
  ),
});

// The full Checkpoints interactive region: run selector + live threshold editor +
// gate matrix + lifecycle + blinded-A/B + promote/deploy. Owns the shared selected-run
// state the compiled page threads through several cards; the gate math runs server-side.
function CheckpointConsoleImpl() {
  const caps = useCaps();
  const [policy, setPolicy] = useState<Record<string, Record<string, number>> | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [gate, setGate] = useState<GatePayload | null>(null);
  const [cmp, setCmp] = useState<any>(null);
  const [champions, setChampions] = useState<any>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    getJSON<any>("/api/gates/policy").then((d) => setPolicy(d.policy)).catch(() => {});
    getJSON<any>("/api/lineage/champions").then(setChampions).catch(() => {});
    getJSON<any>("/api/scoreboards/quality")
      .then((q) => {
        const rs = (q.results ?? []).filter((r: any) => r.suites);
        setRuns(rs);
        setRunId((prev) => prev ?? (rs[0]?.id ?? null));
      })
      .catch(() => {});
  }, []);

  const selected = runs.find((r) => r.id === runId);

  useEffect(() => {
    if (!policy || !selected) return;
    postJSON<GatePayload>("/api/gates/evaluate", { suites: selected.suites, thresholds: policy })
      .then(setGate)
      .catch(() => setGate(null));
  }, [policy, runId, runs]);

  useEffect(() => {
    if (!runId) {
      setCmp(null);
      return;
    }
    getJSON<any>(`/api/comparisons/metrics?candidate_run_id=${encodeURIComponent(runId)}`)
      .then(setCmp)
      .catch(() => setCmp(null));
  }, [runId]);

  const twChampion = champions?.champions?.twotower;

  return (
    <>
      <Card
        title="Configurable ship gates"
        right={
          <select
            value={runId ?? ""}
            onChange={(e) => setRunId(e.target.value)}
            style={{ background: "var(--bg-2)", color: "var(--text)", border: "1px solid var(--border-strong)", borderRadius: "4px", padding: "0.3rem" }}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} · {r.run_id}
              </option>
            ))}
          </select>
        }
      >
        {!selected && <Empty>No experiment with suite metrics available.</Empty>}
        {selected && (
          <div className="two-col">
            <div>
              <div className="hint" style={{ marginBottom: "0.5rem" }}>
                Edit per-suite minimum thresholds — the matrix re-evaluates live via the same{" "}
                <span className="mono">evaluate_ship_gates</span> the ship pipeline uses.
              </div>
              {policy && <ThresholdEditorC policy={policy} onChange={setPolicy} />}
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.5rem" }}>
                <span className="hint">Overall</span>
                {gate && <StatusPillC value={!!gate.pass} label={gate.pass ? "GATES PASS" : "GATES FAIL"} />}
              </div>
              {gate ? <GateMatrixC gate={gate} /> : <Empty>Evaluating…</Empty>}
              {gate?.failures && gate.failures.length > 0 && (
                <ul className="hint" style={{ marginTop: "0.5rem", paddingLeft: "1.1rem" }}>
                  {gate.failures.slice(0, 6).map((ff, i) => (
                    <li key={i} className="mono">
                      {ff}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </Card>

      <div className="two-col">
        <Card title="Lifecycle">
          <TimelineC state={twChampion ? "champion" : selected ? "screened" : "running"} />
          <p className="hint" style={{ marginTop: "0.5rem" }}>
            Lineage lifecycle: running → screened → validated → champion → deployed.
          </p>
        </Card>

        <Card title="Deployment gate (blinded A/B)">
          {cmp ? (
            <>
              <GridC min="120px">
                <StatTileC label="Comparisons" value={cmp.total} sub="need ≥100" accent={cmp.checks?.min_comparisons ? "passed" : undefined} />
                <StatTileC label="Win rate" value={pct(cmp.win_rate)} sub="need >55%" accent={cmp.checks?.win_rate_gt_55 ? "passed" : undefined} />
                <StatTileC label="Wilson LB" value={fmt(cmp.wilson_lower_bound, 2)} sub="need >0.50" accent={cmp.checks?.wilson_gt_50 ? "passed" : undefined} />
              </GridC>
              <StatusPillC value={!!cmp.deployment_ready} label={cmp.deployment_ready ? "deployment ready" : "not ready"} />
            </>
          ) : (
            <Empty>No comparisons for this candidate yet.</Empty>
          )}
        </Card>
      </div>

      <Card title="Promote / Deploy">
        <div className="hint" style={{ marginBottom: "0.7rem" }}>
          These run <span className="mono">model_cycle</span>, which re-checks <span className="mono">promotion_failures</span> /
          <span className="mono"> deployment_failures</span> server-side — the UI never bypasses the real gate.
        </div>
        {jobDef(caps, "model_cycle") ? (
          <div className="two-col">
            <JobLauncherC jobDef={jobDef(caps, "model_cycle")!} execution={caps.execution} defaults={{ subcommand: "promote", run_id: selected?.run_id ?? "", track: "twotower" }} onLaunched={setJobId} />
            <JobLauncherC jobDef={jobDef(caps, "model_cycle")!} execution={caps.execution} defaults={{ subcommand: "deploy", run_id: selected?.run_id ?? "", track: "twotower" }} onLaunched={setJobId} />
          </div>
        ) : (
          <Empty>Promotion actions require the local control plane.</Empty>
        )}
        <LogStreamC jobId={jobId} />
      </Card>
    </>
  );
}

const CheckpointConsole = defineComponent({
  name: "CheckpointConsole",
  description: "Checkpoints interactive region: run selector, live ship-gate editor/matrix, lifecycle, blinded A/B, promote/deploy.",
  props: z.object({}),
  component: () => <CheckpointConsoleImpl />,
});

// The annotate playground surface — a deeply imperative widget (prefetch loop,
// keyboard/swipe, live OpenUI preview) rendered verbatim as its compiled component.
const AnnotatePlayground = defineComponent({
  name: "AnnotatePlayground",
  description: "The full annotate playground surface, rendered verbatim (bespoke stateful widget).",
  props: z.object({}),
  component: () => <Playground />,
});

const CUSTOM = [
  Page,
  PageHead,
  Badge,
  Note,
  Pill,
  EmptyNote,
  Panel,
  Grid,
  Row,
  StatTile,
  StatusPill,
  ProvenanceBadge,
  Bars,
  Sparkline,
  Timeline,
  DataTable,
  GateMatrix,
  NavChip,
  Hint,
  JobList,
  JobsBadge,
  DispatchList,
  ChipTabs,
  JobConsole,
  DataBrowser,
  DataGenerator,
  CheckpointConsole,
  AnnotatePlayground,
];

export const dashboardLibrary = createLibrary({
  components: [...Object.values(openuiLibrary.components), ...CUSTOM] as any,
  root: openuiLibrary.root,
  componentGroups: openuiLibrary.componentGroups,
});
