import React from "react";
import { usePoll, postJSON } from "../api";
import { useCaps } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  StatusPill,
  ProvenanceBadge,
  DataTable,
  Bars,
  Empty,
  ErrorNote,
  pct,
} from "../components";

export function Overview({ navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const { data, error } = usePoll<any>("/api/overview", 15000);
  const jobs = usePoll<any>("/api/jobs", caps.execution ? 4000 : 0);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <div className="loading">Loading mission control…</div>;

  const totals = data.experiment_totals ?? { count: 0, passed: 0 };
  const passRate = totals.count ? totals.passed / totals.count : 0;
  const corpus = data.data ?? {};
  const anns = data.annotations ?? {};
  const activeJobs = (jobs.data?.jobs ?? []).filter((j: any) =>
    ["running", "queued"].includes(j.status)
  );
  const dep = data.system?.deployment ?? {};

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Mission Control</h1>
        <p className="page-sub">
          What is actually going on — live jobs, the experiment scoreboard, checkpoint
          readiness, and corpus health, aggregated from the lineage store and committed
          evidence.
        </p>
      </div>

      <Grid min="200px">
        <StatTile
          label="Experiments passing"
          value={`${totals.passed}/${totals.count}`}
          sub={`${pct(passRate)} pass rate`}
          accent="moss"
        />
        <StatTile
          label="Active jobs"
          value={activeJobs.length}
          sub={caps.execution ? "control plane online" : "read-only"}
          accent={activeJobs.length ? "running" : undefined}
        />
        <StatTile
          label="Training records"
          value={corpus.record_count ?? "—"}
          sub={corpus.version ? `train_data/${corpus.version}` : "fixtures (cold start)"}
        />
        <StatTile
          label="Checkpoints"
          value={(data.checkpoints?.checkpoints ?? []).length}
          sub={dep.selected ? "1 deployed" : "none deployed"}
          accent={dep.selected ? "promoted" : undefined}
        />
        <StatTile label="Human feedback" value={anns.feedback ?? 0} sub={`${anns.human_pairs ?? 0} pref pairs`} />
      </Grid>

      <div className="two-col">
        <Card
          title="Live jobs"
          right={caps.execution ? <StatusPill value="running" label="control plane" /> : <span className="prov prov-committed">read-only</span>}
        >
          {!caps.execution && <Empty>Execution disabled — serve locally to run jobs.</Empty>}
          {caps.execution && activeJobs.length === 0 && <Empty>No jobs running.</Empty>}
          {activeJobs.map((j: any) => (
            <div className="job-line" key={j.id} style={{ display: "flex", justifyContent: "space-between", padding: "0.35rem 0", borderBottom: "1px solid var(--border)" }}>
              <span className="mono">{j.job_key}</span>
              <span style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <StatusPill value={j.status} />
                <button className="chip" onClick={() => postJSON(`/api/jobs/${j.id}/cancel`, {})}>cancel</button>
              </span>
            </div>
          ))}
        </Card>

        <Card title="Experiment scoreboards" right={<button className="chip" onClick={() => navigate("/experiments")}>open →</button>}>
          <Bars
            data={(data.scoreboards ?? []).map((s: any) => ({
              label: `${s.kind} (${s.passed}/${s.count})`,
              value: s.count ? s.passed / s.count : 0,
            }))}
          />
          <p className="hint" style={{ marginTop: "0.6rem" }}>Fraction of experiments passing per matrix.</p>
        </Card>
      </div>

      <Card
        title="Checkpoint roster"
        right={<><ProvenanceBadge provenance={data.runs_provenance} /> <button className="chip" onClick={() => navigate("/checkpoints")}>open →</button></>}
      >
        <DataTable
          columns={[
            { key: "role", label: "Role" },
            { key: "run_id", label: "Run" },
            { key: "kind", label: "Kind" },
            { key: "status", label: "Status" },
          ]}
          rows={(data.checkpoints?.checkpoints ?? []).slice(0, 8)}
          render={{
            run_id: (r) => <span className="mono">{r.run_id || "—"}</span>,
            status: (r) => <StatusPill value={r.status} label={r.status?.slice(0, 22) || "—"} />,
          }}
        />
      </Card>

      <div className="two-col">
        <Card title="Corpus health" right={<ProvenanceBadge provenance={corpus.provenance} />}>
          {corpus.stats ? (
            <Bars
              data={[
                { label: "records", value: corpus.stats.record_count ?? 0 },
                { label: "collected", value: corpus.stats.collected_count ?? 0 },
                { label: "quality rejected", value: corpus.stats.quality_rejected ?? 0 },
                { label: "errors", value: corpus.stats.error_count ?? 0 },
              ]}
            />
          ) : (
            <Bars data={Object.entries(corpus.fixture_counts ?? {}).map(([k, v]) => ({ label: k, value: v as number }))} />
          )}
          <button className="chip" style={{ marginTop: "0.6rem" }} onClick={() => navigate("/data")}>manage data →</button>
        </Card>

        <Card title="System">
          <DataTable
            columns={[{ key: "k", label: "Component" }, { key: "v", label: "State" }]}
            rows={[
              { k: "checkpoint bucket", v: data.system?.checkpoint_bucket ?? "—" },
              { k: "deployed model", v: dep.selected ? "selected" : "none" },
              { k: "outputs/ present", v: String(data.system?.outputs_present) },
              { k: "test suites", v: Object.entries(data.test_data?.suites ?? {}).map(([s, n]) => `${s}:${n}`).join("  ") },
            ]}
            render={{ v: (r) => <span className="mono" style={{ fontSize: "0.74rem" }}>{r.v}</span> }}
          />
        </Card>
      </div>
    </div>
  );
}
