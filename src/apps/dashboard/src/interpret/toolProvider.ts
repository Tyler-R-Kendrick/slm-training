// Data adapter for interpreted mode: maps OpenUI Lang Query("name", args) calls to
// the app's /api, reshaping responses into DSL-friendly row-sets so the .openui pages
// stay simple. This is the interpreted-mode analogue of each compiled page's usePoll.
// Numeric cells are pre-formatted to strings here so table precision matches compiled
// mode exactly (the compiled pages format per-column via fmt(v, digits)).
import { getJSON, postJSON } from "../api";
import { fetchHero } from "../hero";
import { metricLabel, smokeGate } from "../metrics";

const pct = (v: number) => `${Math.round((v || 0) * 100)}%`;

// Mirror components.fmt: null/undefined -> "—", integer -> as-is, else toFixed(digits).
function f(v: unknown, digits = 3): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(digits);
  return String(v);
}

type QueryFn = (args: Record<string, unknown>) => Promise<unknown>;

export const toolProvider: Record<string, QueryFn> = {
  // ---- Overview -----------------------------------------------------------
  overview_hero: async () => fetchHero(),
  overview_insights: async () => {
    const d: any = await getJSON("/api/overview");
    const p = d.performance ?? {};
    const s = p.stats ?? {};
    return {
      references: {
        rows: (p.references ?? []).map((r: any) => ({ ...r, card_state: r.status })),
        provenance: p.reference_provenance,
      },
      tiles: {
        rows: [
          { label: "Reference models", value: String(s.reference_models ?? 0), sub: "champions + latest checkpoint", accent: "promoted" },
          { label: "Experiments reviewed", value: String(s.experiments ?? 0), sub: "quality and grammar evidence", accent: "" },
          { label: "Gate clears", value: `${s.passing ?? 0}/${s.experiments ?? 0}`, sub: "recorded experiment policy", accent: "passed" },
          { label: "Comparable deltas", value: String(s.comparable ?? 0), sub: s.comparable ? "against current reference" : "reference eval required", accent: s.comparable ? "moss" : "failed" },
        ],
      },
      insights: p.insights ?? { improvements: [], carry_forward: [], novel: [] },
      // Metric columns come from the server's ship-gate policy, so a lever
      // change (add/drop a gate metric) re-shapes this table automatically.
      comparisons: {
        columns: [
          { key: "id", label: "Experiment" },
          { key: "run_id", label: "Run" },
          { key: "matrix", label: "Matrix" },
          { key: "gate_status", label: "Gate" },
          ...(p.metric_columns ?? []).map((c: any) => ({ key: c.key, label: c.label, align: "right" })),
          { key: "vs_reference", label: "Vs reference", align: "right" },
        ],
        rows: (p.comparisons ?? []).map((r: any) => ({
          ...r,
          ...Object.fromEntries(
            (p.metric_columns ?? []).map((c: any) => [
              c.key,
              r.metrics?.[c.key] == null ? "—" : pct(r.metrics[c.key]),
            ]),
          ),
        })),
      },
      comparison_basis: p.comparison_basis ?? "",
      cache: p.cache ?? {},
    };
  },
  overview_tiles: async () => {
    const [d, jobsResp]: any = await Promise.all([getJSON("/api/overview"), getJSON("/api/jobs").catch(() => ({ jobs: [] }))]);
    const t = d.experiment_totals ?? { count: 0, passed: 0 };
    const corpus = d.data ?? {};
    const anns = d.annotations ?? {};
    const dep = d.system?.deployment ?? {};
    const active = (jobsResp.jobs ?? []).filter((j: any) => ["running", "queued"].includes(j.status)).length;
    return {
      rows: [
        { label: "Experiments passing", value: `${t.passed}/${t.count}`, sub: `${pct(t.count ? t.passed / t.count : 0)} pass rate`, accent: "moss" },
        { label: "Active jobs", value: String(active), sub: jobsResp.execution ? "control plane online" : "read-only", accent: active ? "running" : "" },
        { label: "Training records", value: String(corpus.record_count ?? "—"), sub: corpus.version ? `train_data/${corpus.version}` : "fixtures (cold start)", accent: "" },
        { label: "Checkpoints", value: String((d.checkpoints?.checkpoints ?? []).length), sub: dep.selected ? "1 deployed" : "none deployed", accent: dep.selected ? "promoted" : "" },
        { label: "Human feedback", value: String(anns.feedback ?? 0), sub: `${anns.human_pairs ?? 0} pref pairs`, accent: "" },
      ],
    };
  },
  overview_jobs: async () => {
    const d: any = await getJSON("/api/jobs").catch(() => ({ jobs: [] }));
    const active = (d.jobs ?? []).filter((j: any) => ["running", "queued"].includes(j.status));
    return { rows: active.map((j: any) => ({ id: j.id, job: j.job_key, status: j.status })), count: active.length, execution: !!d.execution };
  },
  // Raw hub payload; OtelRunList/OtelBadges delegate to the shared views, so
  // compiled and interpreted format identically from the same shape.
  otel_active_runs: async () =>
    getJSON("/api/otel/runs").catch(() => ({ enabled: false, runs: [], peers: [] })),
  overview_scoreboards: async () => {
    const d: any = await getJSON("/api/scoreboards");
    return {
      rows: (d.scoreboards ?? []).map((s: any) => ({
        label: `${s.kind} (${s.passed}/${s.count})`,
        value: s.count ? s.passed / s.count : 0,
      })),
    };
  },
  overview_corpus: async () => {
    const d: any = await getJSON("/api/data/train");
    const stats = d.stats;
    const rows = stats
      ? [
          { label: "records", value: stats.record_count ?? 0 },
          { label: "collected", value: stats.collected_count ?? 0 },
          { label: "quality rejected", value: stats.quality_rejected ?? 0 },
          { label: "errors", value: stats.error_count ?? 0 },
        ]
      : Object.entries(d.fixture_counts ?? {}).map(([label, value]) => ({ label, value }));
    return { rows, provenance: d.provenance };
  },
  overview_system: async () => {
    const d: any = await getJSON("/api/overview");
    const dep = d.system?.deployment ?? {};
    return {
      rows: [
        { k: "checkpoint bucket", v: d.system?.checkpoint_bucket ?? "—" },
        { k: "deployed model", v: dep.selected ? "selected" : "none" },
        { k: "outputs/ present", v: String(d.system?.outputs_present) },
        { k: "test suites", v: Object.entries(d.test_data?.suites ?? {}).map(([s, n]) => `${s}:${n}`).join("  ") },
      ],
    };
  },
  checkpoints_roster: async () => {
    const [ck, ov]: any = await Promise.all([getJSON("/api/checkpoints"), getJSON("/api/overview").catch(() => ({}))]);
    return {
      rows: (ck.checkpoints ?? [])
        .slice(0, 8)
        .map((c: any) => ({ role: c.role, run_id: c.run_id || "—", kind: c.kind, status: c.status })),
      provenance: ov.runs_provenance,
    };
  },
  dispatches: async () => {
    const d: any = await getJSON("/api/dispatches");
    return {
      rows: (d.jobs ?? []).map((j: any) => ({ id: j.id, job: j.job_key, status: j.status, url: j.remote_url })),
      remotes: (d.remotes ?? []).map((r: any) => ({ run_id: r.run_id, url: r.url })),
      bucket_url: d.bucket_url,
    };
  },

  // ---- Experiments --------------------------------------------------------
  scoreboard: async (args) => {
    const kind = String(args.kind || "quality");
    const d: any = await getJSON(`/api/scoreboards/${kind}`);
    const rows = d.results ?? [];
    const metricColumns = d.metric_columns ?? [];
    const passed = rows.filter((r: any) => r.pass === true).length;
    const meta = d.meta ?? {};
    return {
      provenance: d.provenance,
      reference: d.reference ?? "",
      kind,
      meta: {
        count: rows.length,
        passed: `${passed}/${rows.length}`,
        all_pass: rows.length > 0 && passed === rows.length,
        matrix: meta.matrix_set ?? meta.matrix ?? kind,
        steps: meta.steps ?? "—",
      },
      columns: [
        { key: "id", label: "id" },
        { key: "date", label: "date" },
        { key: "description", label: "experiment" },
        { key: "pass_status", label: "gate" },
        ...metricColumns.map((c: any) => ({ key: c.key, label: c.label, align: "right" })),
        { key: "agentv", label: "AgentV", align: "right" },
        { key: "trace", label: "trace", align: "right" },
      ],
      rows: rows.map((r: any) => ({
        id: r.id,
        run_id: r.run_id || r.id,
        date: r.date || "—",
        description: (r.description || "").slice(0, 70),
        pass_status: r.pass === undefined ? "" : r.pass ? "pass" : "fail",
        // Server-normalized suites; the guarded legacy fallback arrives tagged
        // (meaningful_source) and renders with the same * marker as compiled.
        ...Object.fromEntries(metricColumns.map((c: any) => {
          const values = r.suites?.[c.suite] ?? {};
          const v = values[c.metric];
          if (v === undefined || v === null) return [c.key, "—"];
          const legacy = c.metric === "meaningful_program_rate" && values.meaningful_source === "parse_rate_legacy";
          return [c.key, `${f(v, 2)}${legacy ? "*" : ""}`];
        })),
        agentv: r.agentv?.total === undefined ? "—" : `${r.agentv.passed ?? 0}/${r.agentv.total}`,
        trace: r.trace_id ? String(r.trace_id).slice(0, 12) : "—",
      })),
    };
  },

  // ---- Smoke --------------------------------------------------------------
  smoke_perf: async () => {
    const d: any = await getJSON("/api/scoreboards/perf");
    const rows = d.results ?? [];
    const first = rows[0] ?? {};
    const ph = first.phase_summary ?? {};
    const runId = first.run_id || first.id || "";
    const detail: any = runId ? await getJSON(`/api/runs/${encodeURIComponent(runId)}`).catch(() => ({})) : {};
    const phases = detail.insights?.phases;
    return {
      provenance: d.provenance,
      first_id: runId,
      tiles: [
        { label: "Latency p50", value: first.latency_ms_p50 ? `${f(first.latency_ms_p50, 0)}ms` : "—", accent: "ember" },
        { label: "Latency p95", value: first.latency_ms_p95 ? `${f(first.latency_ms_p95, 0)}ms` : "—", accent: "" },
        { label: "Tokens/sec", value: f(first.tokens_per_sec, 1), accent: "moss" },
        { label: "Wall sec", value: f(first.wall_sec, 2), accent: "" },
        { label: "Perf runs", value: String(rows.length), accent: "" },
      ],
      phase: phases?.length ? phases : [
        { label: "denoiser", value: ph.denoiser_ms_mean ?? 0, help: "Profile model forwards first; test AMP/compile, batching, or fewer decode steps, then rerun quality guardrails." },
        { label: "dfa sync", value: ph.dfa_sync_ms_mean ?? 0, help: "Remove unnecessary host/device synchronization boundaries and measure again on the same device." },
        { label: "stream check", value: ph.stream_check_ms_mean ?? 0, help: "Prefer incremental or chosen-token verification; keep final validation and parse guardrails enabled." },
      ],
      rows: rows.map((r: any) => ({
        id: r.id,
        run_id: r.run_id || r.id,
        latency_ms_p50: r.latency_ms_p50,
        tokens_per_sec: r.tokens_per_sec,
        parse_rate: r.parse_rate,
        guardrails: r.guardrails,
      })),
    };
  },
  smoke_quality: async () => {
    // The canary gate threshold comes from the live ship-gate policy, not a
    // hardcoded 0.66 — if the smoke lever moves, the pass/fail pill follows.
    const [d, pol]: any[] = await Promise.all([
      getJSON("/api/scoreboards/quality"),
      getJSON("/api/gates/policy").catch(() => ({ policy: {} })),
    ]);
    const { lever, threshold, label } = smokeGate(pol.policy);
    return {
      provenance: d.provenance,
      gate_label: label,
      headline_label: metricLabel(lever),
      rows: (d.results ?? []).map((r: any) => {
        // Server-normalized suites; no client parse_rate substitution — the
        // guarded legacy fallback arrives tagged and renders with a * marker.
        const suite = r.suites?.smoke ?? {};
        const headline = suite[lever];
        const legacy = suite.meaningful_source === "parse_rate_legacy";
        const ci = suite[`${lever}_ci95`];
        const ciSuffix =
          headline != null && Array.isArray(ci) ? ` [${f(ci[0], 2)}, ${f(ci[1], 2)}]` : "";
        return {
          id: r.id,
          run_id: r.run_id || r.id,
          parse: `${f(headline, 2)}${headline != null && legacy ? "*" : ""}${ciSuffix}`,
          fidelity: f(suite.placeholder_fidelity, 2),
          reward: f(suite.reward_score, 2),
          parse_status:
            headline == null || threshold == null
              ? ""
              : headline >= threshold
                ? "pass"
                : "fail",
        };
      }),
    };
  },

  // ---- Data ---------------------------------------------------------------
  data_train: async (args) => {
    const version = args.version ? `?version=${encodeURIComponent(String(args.version))}` : "";
    const d: any = await getJSON(`/api/data/train${version}`);
    const stats = d.stats;
    return {
      provenance: d.provenance,
      version: d.version,
      versions: d.versions ?? [],
      tiles: [
        { label: "Records", value: String(d.record_count ?? stats?.record_count ?? "—"), sub: d.version === "examples" ? "built-in examples" : d.path ?? "no data", accent: "moss" },
        { label: "Storage", value: String(d.storage ?? "committed"), sub: d.fingerprint ? String(d.fingerprint).slice(0, 12) : null, accent: "" },
        { label: "Build trace", value: d.trace_id ? String(d.trace_id).slice(0, 12) : "—", sub: d.trace_id ? "W3C trace ID" : null, accent: "" },
        { label: "Collected", value: String(stats?.collected_count ?? "—"), sub: null, accent: "" },
        { label: "Quality rejected", value: String(stats?.quality_rejected ?? "—"), sub: null, accent: stats?.quality_rejected ? "failed" : "" },
        { label: "Synthesizer", value: String(stats?.synthesizer ?? "—"), sub: null, accent: "" },
        { label: "Errors", value: String(stats?.error_count ?? "—"), sub: null, accent: stats?.error_count ? "failed" : "" },
        { label: "Profile", value: String(d.profile ?? "—"), sub: d.quality ? "curation profile" : null, accent: "" },
        { label: "Redundancy dropped", value: String(d.quality?.redundancy_dropped ?? "—"), sub: null, accent: d.quality?.redundancy_dropped ? "ember" : "" },
        { label: "Decontam flagged", value: String(d.quality?.decontam_flagged ?? "—"), sub: null, accent: d.quality?.decontam_flagged ? "failed" : "" },
      ],
      used_by: d.used_by_runs ?? [],
      composition: stats
        ? [
            { label: "records", value: stats.record_count ?? 0 },
            { label: "collected", value: stats.collected_count ?? 0 },
            { label: "quality rejected", value: stats.quality_rejected ?? 0 },
            { label: "reserved-test rejected", value: stats.structure_reserved_rejected ?? 0 },
          ]
        : Object.entries(d.fixture_counts ?? {}).map(([label, value]) => ({ label, value })),
    };
  },
  data_test: async () => {
    const d: any = await getJSON("/api/data/test");
    return { provenance: d.provenance, rows: Object.entries(d.suites ?? {}).map(([label, value]) => ({ label, value })) };
  },
  data_preference: async () => {
    const d: any = await getJSON("/api/data/preference");
    return { provenance: d.provenance, rows: d.rows ?? [] };
  },
  data_records: async (args) => {
    const v = args.version ? String(args.version) : "";
    if (!v) return { count: 0, rows: [] };
    const d: any = await getJSON(`/api/data/train/${encodeURIComponent(v)}/records?limit=40`).catch(() => ({ records: [], count: 0 }));
    return {
      count: d.count ?? (d.records ?? []).length,
      rows: (d.records ?? []).map((r: any) => ({
        id: r.id,
        split: r.split,
        source: r.source,
        prompt: (r.prompt || "").slice(0, 90),
      })),
    };
  },

  // ---- Checkpoints --------------------------------------------------------
  checkpoints_tiles: async () => {
    const [ck, ch, q]: any = await Promise.all([
      getJSON("/api/checkpoints"),
      getJSON("/api/lineage/champions").catch(() => ({})),
      getJSON("/api/scoreboards/quality").catch(() => ({})),
    ]);
    const dep = ck.deployment ?? {};
    const tw = ch.champions?.twotower;
    const runs = (q.results ?? []).filter((r: any) => r.suites);
    let cmp: any = {};
    if (runs[0]) cmp = await getJSON(`/api/comparisons/metrics?candidate_run_id=${encodeURIComponent(runs[0].id)}`).catch(() => ({}));
    return {
      rows: [
        { label: "Checkpoints", value: String((ck.checkpoints ?? []).length), sub: null, accent: "moss" },
        { label: "Deployed", value: dep.selected ? "1" : "0", sub: dep.selected?.track ?? "none selected", accent: dep.selected ? "promoted" : "" },
        { label: "TwoTower champion", value: tw ? "set" : "none", sub: tw?.run_id ?? "", accent: tw ? "promoted" : "" },
        { label: "A/B comparisons", value: String(cmp.total ?? 0), sub: cmp.total !== undefined ? `${pct(cmp.win_rate)} win` : "", accent: "" },
      ],
    };
  },
  checkpoints_roster_full: async () => {
    const d: any = await getJSON("/api/checkpoints");
    return {
      provenance: "committed",
      rows: (d.checkpoints ?? []).map((c: any) => ({
        role: c.role,
        run_id: c.run_id || "—",
        architecture: c.architecture,
        parameters: c.parameters,
        model_size: c.model_size,
        throughput: c.throughput,
        status: c.status,
      })),
    };
  },
  gates_policy: async () => {
    const d: any = await getJSON("/api/gates/policy");
    return d;
  },
  gates_evaluate: async (args) => {
    return postJSON("/api/gates/evaluate", { suites: args.suites, thresholds: args.thresholds });
  },
  quality_runs: async () => {
    const d: any = await getJSON("/api/scoreboards/quality");
    return { rows: (d.results ?? []).filter((r: any) => r.suites).map((r: any) => ({ id: r.id, run_id: r.run_id, suites: r.suites })) };
  },
  champions: async () => getJSON("/api/lineage/champions").catch(() => ({})),
  comparison_metrics: async (args) => {
    const rid = String(args.candidate_run_id || "");
    if (!rid) return {};
    return getJSON(`/api/comparisons/metrics?candidate_run_id=${encodeURIComponent(rid)}`);
  },
};
