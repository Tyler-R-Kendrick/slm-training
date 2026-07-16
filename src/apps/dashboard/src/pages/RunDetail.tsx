import React from "react";
import { postJSON, usePoll } from "../api";
import { useCaps } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  StatusPill,
  GateMatrix,
  Timeline,
  Bars,
  DataTable,
  ProvenanceBadge,
  Empty,
  ErrorNote,
  fmt,
} from "../components";

type LossPoint = { step: number; loss: number | null };
type LossEvent = { kind: string; step: number; loss: number | null; severity: string; finding: string; suggestion: string };
const BROWSER_INFERENCE_MODULE = "/static/browser_inference.js";

function LossChart({ points, events }: { points: LossPoint[]; events: LossEvent[] }) {
  const finite = points.filter((point): point is { step: number; loss: number } => Number.isFinite(point.loss));
  if (finite.length < 2) return <Empty>No loss series was recorded for this run.</Empty>;
  const width = 900;
  const height = 260;
  const pad = { left: 58, right: 20, top: 20, bottom: 38 };
  const xMin = Math.min(...finite.map((point) => point.step));
  const xMax = Math.max(...finite.map((point) => point.step));
  const yMin = Math.min(...finite.map((point) => point.loss));
  const yMax = Math.max(...finite.map((point) => point.loss));
  const x = (step: number) => pad.left + ((step - xMin) / (xMax - xMin || 1)) * (width - pad.left - pad.right);
  const y = (loss: number) => height - pad.bottom - ((loss - yMin) / (yMax - yMin || 1)) * (height - pad.top - pad.bottom);
  const path = finite.map((point, index) => `${index ? "L" : "M"}${x(point.step).toFixed(1)},${y(point.loss).toFixed(1)}`).join(" ");
  return (
    <div className="loss-chart-wrap">
      <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Loss over ${finite.length} recorded points with ${events.length} collapse indicators`}>
        {[0, 0.5, 1].map((fraction) => {
          const value = yMin + (yMax - yMin) * (1 - fraction);
          const cy = pad.top + fraction * (height - pad.top - pad.bottom);
          return <g key={fraction}><line x1={pad.left} y1={cy} x2={width - pad.right} y2={cy} className="loss-grid" /><text x={pad.left - 8} y={cy + 4} textAnchor="end">{fmt(value, 3)}</text></g>;
        })}
        <path d={path} className="loss-line" />
        {events.filter((event) => Number.isFinite(event.loss)).map((event, index) => (
          <g key={`${event.kind}-${event.step}-${index}`} className={`loss-marker loss-marker-${event.severity}`}>
            <line x1={x(event.step)} y1={pad.top} x2={x(event.step)} y2={height - pad.bottom} />
            <circle cx={x(event.step)} cy={y(Number(event.loss))} r="6"><title>{`Step ${event.step}: ${event.finding}`}</title></circle>
          </g>
        ))}
        <text x={width / 2} y={height - 8} textAnchor="middle">step</text>
        <text x="14" y={height / 2} textAnchor="middle" transform={`rotate(-90 14 ${height / 2})`}>loss</text>
      </svg>
    </div>
  );
}

function storedBool(key: string, fallback: boolean) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value === "true";
  } catch {
    return fallback;
  }
}

function InsightEnrichment({ report, completed, reload }: { report: any; completed: boolean; reload: () => void }) {
  const caps = useCaps();
  const [enabled, setEnabled] = React.useState(() => storedBool("openui.runInsights.enabled", true));
  const [openaiFallback, setOpenaiFallback] = React.useState(() => storedBool("openui.runInsights.openaiFallback", false));
  const [status, setStatus] = React.useState("");
  const [sessionGenerated, setSessionGenerated] = React.useState<any>(null);
  const attempted = React.useRef("");

  const setPreference = (key: string, value: boolean, setter: (next: boolean) => void) => {
    setter(value);
    try { localStorage.setItem(key, String(value)); } catch { /* storage can be unavailable */ }
  };

  const generate = React.useCallback(async () => {
    if (!report?.source_fingerprint || !caps.execution) return;
    setStatus("Starting on-device analysis…");
    let session: any = null;
    let generatedForSession: any = null;
    try {
      const browser: any = await import(/* @vite-ignore */ BROWSER_INFERENCE_MODULE);
      const created = await browser.createBrowserModelSession({
        mode: "insights",
        onProgress: ({ status: next, progress }: any) => setStatus(`${next}${Number.isFinite(progress) ? ` ${Math.round(progress * 100)}%` : ""}`),
      });
      session = created.session;
      const raw = await session.prompt(browser.buildRunInsightsPrompt(report));
      const generated = browser.parseRunInsightsResponse(raw);
      generatedForSession = generated;
      setSessionGenerated(generated);
      setStatus("Persisting browser analysis…");
      await postJSON(`/api/runs/${encodeURIComponent(report.run_id)}/insights`, {
        source_fingerprint: report.source_fingerprint,
        provider: "browser",
        runtime: created.runtime,
        generated,
      });
      setSessionGenerated(null);
      setStatus("Browser analysis persisted as autoresearch evidence.");
      reload();
    } catch (error: any) {
      if (openaiFallback && caps.run_insights?.openai_available) {
        try {
          setStatus("Browser analysis unavailable; trying the opt-in OpenAI fallback…");
          await postJSON(`/api/runs/${encodeURIComponent(report.run_id)}/insights/openai`, {});
          setSessionGenerated(null);
          setStatus("OpenAI analysis persisted as autoresearch evidence.");
          reload();
          return;
        } catch (fallbackError: any) {
          setStatus(`Analysis failed: ${fallbackError?.message || fallbackError}`);
          return;
        }
      }
      setStatus(`Browser analysis unavailable: ${error?.message || error}${generatedForSession ? " (result remains session-only)" : ""}`);
    } finally {
      session?.destroy?.();
    }
  }, [caps.execution, caps.run_insights?.openai_available, openaiFallback, reload, report]);

  React.useEffect(() => {
    if (!enabled || !completed || report?.enrichment || !caps.execution) return;
    if (attempted.current === report?.source_fingerprint) return;
    attempted.current = report?.source_fingerprint;
    void generate();
  }, [caps.execution, completed, enabled, generate, report?.enrichment, report?.source_fingerprint]);

  const generated = report?.enrichment?.generated ?? sessionGenerated;
  return (
    <Card title="Generated diagnosis" right={report?.enrichment ? <span className="prov prov-live">persisted</span> : sessionGenerated ? <span className="pill pill-warning">session only</span> : null}>
      <div className="insight-settings">
        <label><input type="checkbox" checked={enabled} onChange={(event) => setPreference("openui.runInsights.enabled", event.target.checked, setEnabled)} /> browser insights</label>
        <label title="Used only when browser inference fails and the server has OPENAI_API_KEY configured."><input type="checkbox" checked={openaiFallback} onChange={(event) => setPreference("openui.runInsights.openaiFallback", event.target.checked, setOpenaiFallback)} /> OpenAI fallback (opt-in)</label>
        {caps.execution && !report?.enrichment && <button className="btn btn-small" onClick={() => { attempted.current = ""; void generate(); }}>Analyze now</button>}
      </div>
      {status && <p className="hint" aria-live="polite">{status}</p>}
      {generated ? <div className="generated-insights"><p>{generated.summary}</p>{generated.causes?.map((cause: any, index: number) => <div className="insight-cause" key={index}><strong>{cause.title}</strong> <span className="hint">{fmt(cause.confidence * 100, 0)}% hypothesis confidence</span><p>{cause.rationale}</p><p className="hint">Try: {cause.suggestion}</p></div>)}{generated.phase_suggestions?.map((item: any, index: number) => <div className="insight-cause" key={`phase-${index}`}><strong>{item.phase}</strong><p className="hint">Try: {item.suggestion}</p></div>)}</div> : <Empty>Deterministic indicators remain available even when optional AI analysis is disabled or unavailable.</Empty>}
    </Card>
  );
}

export function RunDetail({ runId, navigate }: { runId: string; navigate: (to: string) => void }) {
  const { data, error, reload } = usePoll<any>(`/api/runs/${encodeURIComponent(runId)}`, 0);
  const traceLimit = 20;
  const [traceOffset, setTraceOffset] = React.useState(0);
  const {
    data: traceData,
    error: traceError,
    loading: tracesLoading,
  } = usePoll<any>(
    `/api/runs/${encodeURIComponent(runId)}/rl-traces?offset=${traceOffset}&limit=${traceLimit}`,
    0,
  );
  React.useEffect(() => setTraceOffset(0), [runId]);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <div className="loading">Loading run…</div>;

  const ts = data.train_summary ?? {};
  const track = ts.track ?? {};
  const sb = data.scoreboard ?? {};
  const telem = data.telemetry ?? ts.telemetry ?? {};
  const spans = telem.spans ?? {};
  const bucket = ts.checkpoint_bucket ?? {};
  const finalEval = ts.final_eval?.suites ?? sb.suites ?? {};
  const report = data.insights ?? {};
  const loss = report.loss ?? {};

  const spanBars = (report.phases?.length ? report.phases : Object.entries(spans)
    .map(([name, v]: [string, any]) => ({ label: name, value: Number(v?.pct ?? 0) })))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const evalRows = Object.entries(finalEval).map(([suite, m]: [string, any]) => ({
    suite,
    parse_rate: m?.parse_rate,
    structural_similarity: m?.structural_similarity,
    placeholder_fidelity: m?.placeholder_fidelity,
    reward_score: m?.reward_score,
    n: m?.n,
  }));

  return (
    <div>
      <div className="page-head">
        <a className="chip" onClick={() => navigate("/experiments")} style={{ marginBottom: "0.6rem", display: "inline-block" }}>← back</a>
        <h1 className="page-title" style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
          <span className="mono">{runId}</span>
          <ProvenanceBadge provenance={data.provenance} />
        </h1>
        <p className="page-sub">{sb.description || ts.model || "Run detail"}</p>
      </div>

      <Grid min="180px">
        <StatTile label="Trace ID" value={data.trace?.trace_id?.slice(0, 12) ?? data.manifest?.trace_id?.slice(0, 12) ?? "—"} sub={data.trace?.trace_id ? "W3C correlated" : undefined} />
        <StatTile label="Steps" value={fmt(ts.steps)} accent="moss" sub={ts.stopped_on ? `stop: ${ts.stopped_on}` : undefined} />
        <StatTile label="Last loss" value={fmt(ts.last_loss, 4)} />
        <StatTile label="Best weighted NLL" value={fmt(ts.best_weighted_nll, 4)} accent="ember" />
        <StatTile label="Best ship score" value={fmt(ts.best_ship_score, 3)} />
        <StatTile label="Records" value={fmt(ts.record_count ?? sb.n)} />
        <StatTile label="Gate" value={data.gates?.pass === undefined ? "—" : ""} sub={sb.matrix ? `${sb.matrix} matrix` : undefined}
          accent={data.gates?.pass ? "passed" : data.gates ? "failed" : undefined} />
      </Grid>

      <Card title="Loss over time" right={<StatusPill value={loss.status ?? "unavailable"} label={loss.status ?? "unavailable"} />}>
        <LossChart points={loss.points ?? []} events={loss.events ?? []} />
        {!!loss.events?.length && <div className="collapse-findings">{loss.events.map((event: LossEvent, index: number) => <details key={`${event.kind}-${event.step}-${index}`}><summary><StatusPill value={event.severity === "critical" ? false : "warning"} label={`${event.kind.replaceAll("_", " ")} · step ${event.step}`} /></summary><p>{event.finding}</p><p className="hint">Suggested experiment: {event.suggestion}</p></details>)}</div>}
      </Card>

      <InsightEnrichment report={report} completed={Boolean(ts.finished_at || data.manifest?.lifecycle_state === "screened" || sb.id)} reload={reload} />

      <div className="two-col">
        <Card title="Lifecycle">
          <Timeline state={data.manifest?.lifecycle_state ?? (sb.pass ? "screened" : "running")} />
          <DataTable
            columns={[{ key: "k", label: "Track" }, { key: "v", label: "" }]}
            rows={[
              { k: "context backend", v: track.context_backend ?? "—" },
              { k: "trainable params", v: fmt(track.trainable_params) },
              { k: "frozen params", v: fmt(track.frozen_params) },
              { k: "tokens / param", v: fmt(track.tokens_per_trainable_param, 2) },
              { k: "output tokenizer", v: track.output_tokenizer ?? "—" },
              { k: "finished", v: ts.finished_at ?? "—" },
            ]}
            render={{ v: (r) => <span className="mono" style={{ fontSize: "0.76rem" }}>{r.v}</span> }}
          />
          {bucket.bucket_url && (
            <a className="chip active" href={bucket.bucket_url} target="_blank" rel="noreferrer" style={{ marginTop: "0.6rem", display: "inline-block" }}>
              durable checkpoint ↗
            </a>
          )}
        </Card>

        <Card title="Ship gates" right={data.gates && <StatusPill value={!!data.gates.pass} label={data.gates.pass ? "pass" : "fail"} />}>
          {data.gates ? <GateMatrix gate={data.gates} /> : <Empty>No gate evidence for this run.</Empty>}
        </Card>
      </div>

      <div className="two-col">
        <Card title="Telemetry (share of cycle)">
          {spanBars.length ? <Bars data={spanBars} /> : <Empty>No telemetry for this run.</Empty>}
        </Card>
        <Card title="Eval suites">
          {evalRows.length ? (
            <DataTable
              columns={[
                { key: "suite", label: "suite" },
                { key: "parse_rate", label: "parse", align: "right" },
                { key: "structural_similarity", label: "struct", align: "right" },
                { key: "placeholder_fidelity", label: "fidelity", align: "right" },
                { key: "reward_score", label: "reward", align: "right" },
                { key: "n", label: "n", align: "right" },
              ]}
              rows={evalRows}
              render={{
                suite: (r) => <span className="mono">{r.suite}</span>,
                parse_rate: (r) => fmt(r.parse_rate, 2),
                structural_similarity: (r) => fmt(r.structural_similarity, 2),
                placeholder_fidelity: (r) => fmt(r.placeholder_fidelity, 2),
                reward_score: (r) => fmt(r.reward_score, 2),
              }}
            />
          ) : (
            <Empty>No eval scoreboard for this run.</Empty>
          )}
        </Card>
      </div>

      <Card
        title="RL traces"
        right={traceData && <span className="hint">{fmt(traceData.total)} normalized traces</span>}
      >
        {traceError ? (
          <ErrorNote error={traceError} />
        ) : !traceData ? (
          <div className="loading">Loading RL traces…</div>
        ) : traceData.traces.length === 0 ? (
          <Empty>No normalized RL traces for this run.</Empty>
        ) : (
          <div>
            {traceData.invalid_rows > 0 && (
              <div className="error-note">Skipped {traceData.invalid_rows} malformed trace rows.</div>
            )}
            {traceData.traces.map((trace: any, index: number) => (
              <details className="record-detail" key={`${trace.step}-${trace.rollout_id}-${index}`}>
                <summary>
                  <span className="mono">{trace.engine} · step {trace.step} · {trace.rollout_id || "rollout"}</span>
                  {trace.truncated && <StatusPill value={false} label="truncated" />}
                </summary>
                <div className="chip-row" style={{ marginTop: "0.7rem" }}>
                  <span className="chip">group {trace.group_id || "—"}</span>
                  <span className="chip">reward {fmt(trace.rewards?.composite, 4)}</span>
                  <span className="chip">parse {fmt(trace.rewards?.parse, 4)}</span>
                  <span className="chip">slots {fmt(trace.rewards?.placeholder_fidelity, 4)}</span>
                  <span className="chip">structure {fmt(trace.rewards?.structural_similarity, 4)}</span>
                  <span className="chip">tokens {fmt(trace.prompt_tokens)} + {fmt(trace.completion_tokens)}</span>
                </div>
                <h3>Prompt</h3>
                <pre>{trace.prompt}</pre>
                <h3>Completion</h3>
                <pre>{trace.completion}</pre>
                <h3>Gold OpenUI</h3>
                <pre>{trace.gold_openui}</pre>
                <h3>Action token IDs</h3>
                <pre>{JSON.stringify(trace.action_token_ids)}</pre>
                {trace.rollout_logprobs && (
                  <>
                    <h3>Rollout log probabilities</h3>
                    <pre>{JSON.stringify(trace.rollout_logprobs)}</pre>
                  </>
                )}
              </details>
            ))}
            <div className="data-browser-pages">
              <button
                className="btn btn-small"
                disabled={tracesLoading || traceOffset === 0}
                onClick={() => setTraceOffset(Math.max(0, traceOffset - traceLimit))}
              >
                Previous
              </button>
              <span className="hint">
                {traceOffset + 1}–{Math.min(traceOffset + traceData.count, traceData.total)} of {traceData.total}
              </span>
              <button
                className="btn btn-small"
                disabled={tracesLoading || traceOffset + traceData.count >= traceData.total}
                onClick={() => setTraceOffset(traceOffset + traceLimit)}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
