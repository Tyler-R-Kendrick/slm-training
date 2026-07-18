import React, { useEffect, useState } from "react";
import { getJSON, postJSON, usePoll } from "../api";
import { useCaps, jobDef } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  DataTable,
  Bars,
  LogStream,
  ProvenanceBadge,
  ErrorNote,
} from "../components";

const FETCH_SIZE = 500;

export function TrainingDataBrowser({ version }: { version?: string }) {
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("");
  const [selected, setSelected] = useState<any>(null);
  const [records, setRecords] = useState<any[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [count, setCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSelected(null);
    setQuery("");
    setSource("");
  }, [version]);

  useEffect(() => {
    setSelected(null);
    setRecords([]);
    setCount(0);
    setError(null);
    if (!version) return;

    let cancelled = false;
    setLoading(true);
    (async () => {
      const all: any[] = [];
      let total = 0;
      do {
        const params = new URLSearchParams({ limit: String(FETCH_SIZE), offset: String(all.length) });
        if (query) params.set("q", query);
        if (source) params.set("source", source);
        const page = await getJSON<any>(`/api/data/train/${encodeURIComponent(version)}/records?${params}`);
        if (cancelled) return;
        if (!all.length) setSources(page.sources ?? []);
        total = page.count ?? 0;
        const batch = page.records ?? [];
        all.push(...batch);
        if (!batch.length) break;
      } while (all.length < total);
      if (!cancelled) {
        setRecords(all);
        setCount(total);
      }
    })()
      .catch((e) => !cancelled && setError(String(e?.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [version, query, source]);

  const status = loading ? `Loading ${records.length ? `${records.length} of ${count}` : "records"}…` : `Showing all ${records.length} of ${count}`;

  return (
    <div className="data-browser">
      <div className="data-browser-tools">
        <label className="launcher-field">
          <span>Search records</span>
          <input
            type="search"
            value={query}
            placeholder="id, prompt, source, or OpenUI"
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <label className="launcher-field">
          <span>Source</span>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option>
            {sources.map((item: string) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
        <span className="hint data-browser-count">{status}</span>
      </div>
      <ErrorNote error={error} />
      <DataTable
        columns={[
          { key: "prompt", label: "prompt" },
          { key: "target", label: "expected layout" },
          { key: "source", label: "source" },
          { key: "view", label: "" },
        ]}
        rows={records}
        maxHeight="38rem"
        render={{
          prompt: (r) => <span className="data-prompt">{r.prompt}</span>,
          target: (r) => <span className="data-target mono">{r.openui}</span>,
          view: (r) => (
            <button className="btn btn-small" onClick={() => setSelected(r)}>
              View
            </button>
          ),
        }}
      />
      {selected && (
        <div className="record-detail">
          <div className="record-detail-head">
            <div>
              <strong className="mono">{selected.id}</strong>
              <div className="hint">{selected.source} · {selected.split}</div>
            </div>
            <button className="btn btn-small" onClick={() => setSelected(null)}>Close</button>
          </div>
          <h3>Prompt</h3>
          <p>{selected.prompt}</p>
          <h3>OpenUI target</h3>
          <pre>{selected.openui}</pre>
          {selected.design_md && (
            <>
              <h3>DESIGN.md</h3>
              <pre>{selected.design_md}</pre>
            </>
          )}
          <h3>Complete record</h3>
          <pre>{JSON.stringify(selected, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export function DataQualityCard({
  version,
  usedByRuns = [],
  navigate,
}: {
  version?: string;
  usedByRuns?: string[];
  navigate?: (to: string) => void;
}) {
  const [payload, setPayload] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPayload(null);
    setError(null);
    if (!version || version === "examples") return;
    let cancelled = false;
    getJSON<any>(`/api/data/train/${encodeURIComponent(version)}/quality`)
      .then((d) => !cancelled && setPayload(d))
      .catch((e) => !cancelled && setError(String(e?.message ?? e)));
    return () => { cancelled = true; };
  }, [version]);

  const summary = payload?.summary;
  const stageRows = Object.entries(summary?.rejected_by_stage ?? {}).map(([label, value]) => ({ label, value: value as number }));
  const engines = summary?.engines ?? {};
  return (
    <div className="data-quality">
      <ErrorNote error={error} />
      {!summary ? (
        <p className="hint">
          {version === "examples"
            ? "Built-in examples carry no build-time quality report."
            : "No quality report for this dataset version (rebuild with the strict profile to generate one)."}
        </p>
      ) : (
        <>
          <Grid min="150px">
            <StatTile label="Profile" value={summary.profile ?? "—"} accent="moss" />
            <StatTile label="Admitted" value={summary.admitted ?? "—"} sub={summary.admission_rate != null ? `${Math.round(summary.admission_rate * 100)}% of candidates` : null} />
            <StatTile label="Rejected" value={summary.rejected_total ?? "—"} accent={summary.rejected_total ? "ember" : undefined} sub="see audit ledger" />
            <StatTile label="Parse rate" value={summary.parse_rate != null ? `${Math.round(summary.parse_rate * 100)}%` : "—"} />
            <StatTile label="Judge pass" value={summary.judge_pass_rate != null ? `${Math.round(summary.judge_pass_rate * 100)}%` : "—"} />
            <StatTile label="Mean quality" value={summary.mean_quality_score ?? "—"} />
            <StatTile label="Redundancy dropped" value={summary.redundancy_dropped ?? 0} accent={summary.redundancy_dropped ? "ember" : undefined} />
            <StatTile label="Decontam flagged" value={summary.decontam_flagged ?? 0} accent={summary.decontam_flagged ? "failed" : undefined} />
          </Grid>
          {stageRows.length > 0 && <Bars data={stageRows} />}
          <p className="hint">
            Engines: similarity {engines.similarity ?? "—"}
            {engines.semantic_dedup ? ` · semantic ${engines.semantic_dedup}` : ""}
            {engines.decontam ? ` · decontam ${engines.decontam}` : ""}
          </p>
        </>
      )}
      {usedByRuns.length > 0 && (
        <div className="chip-row" style={{ marginTop: "0.6rem" }}>
          <span className="hint">Used by runs:</span>
          {usedByRuns.map((id) => (
            <span key={id} className="chip" onClick={() => navigate?.(`/runs/${encodeURIComponent(id)}`)}>{id}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export function RejectedRecordsBrowser({ version }: { version?: string }) {
  const LIMIT = 50;
  const [stage, setStage] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<any>(null);
  const [selected, setSelected] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStage("");
    setOffset(0);
    setSelected(null);
  }, [version]);

  useEffect(() => {
    setPage(null);
    setError(null);
    if (!version || version === "examples") return;
    let cancelled = false;
    const params = new URLSearchParams({ limit: String(LIMIT), offset: String(offset) });
    if (stage) params.set("stage", stage);
    getJSON<any>(`/api/data/train/${encodeURIComponent(version)}/rejected?${params}`)
      .then((d) => !cancelled && setPage(d))
      .catch((e) => !cancelled && setError(String(e?.message ?? e)));
    return () => { cancelled = true; };
  }, [version, stage, offset]);

  if (!version || version === "examples") {
    return <p className="hint">Built-in examples carry no rejection ledger.</p>;
  }
  const rows = page?.rejected ?? [];
  return (
    <div className="data-browser">
      <div className="data-browser-tools">
        <label className="launcher-field">
          <span>Stage</span>
          <select value={stage} onChange={(e) => { setOffset(0); setStage(e.target.value); }}>
            <option value="">All stages</option>
            {(page?.stages ?? []).map((item: string) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
        <span className="hint data-browser-count">
          {page ? `${page.count} rejected candidate${page.count === 1 ? "" : "s"}` : "Loading…"}
        </span>
      </div>
      <ErrorNote error={error} />
      {rows.length === 0 && page ? (
        <p className="hint">Nothing was rejected{stage ? ` at stage ${stage}` : ""} — or this version predates the audit ledger.</p>
      ) : (
        <DataTable
          columns={[
            { key: "id", label: "id" },
            { key: "stage", label: "stage" },
            { key: "reason", label: "reason" },
            { key: "view", label: "" },
          ]}
          rows={rows}
          maxHeight="24rem"
          render={{
            id: (r) => <span className="mono">{r.id}</span>,
            view: (r) => (
              <button className="btn btn-small" onClick={() => setSelected(r)}>View</button>
            ),
          }}
        />
      )}
      {page && page.count > LIMIT && (
        <div className="data-browser-pages">
          <button className="btn btn-small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>Previous</button>
          <span className="hint">{offset + 1}–{Math.min(offset + LIMIT, page.count)} of {page.count}</span>
          <button className="btn btn-small" disabled={offset + LIMIT >= page.count} onClick={() => setOffset(offset + LIMIT)}>Next</button>
        </div>
      )}
      {selected && (
        <div className="record-detail">
          <div className="record-detail-head">
            <div>
              <strong className="mono">{selected.id}</strong>
              <div className="hint">{selected.stage} · {selected.reason}</div>
            </div>
            <button className="btn btn-small" onClick={() => setSelected(null)}>Close</button>
          </div>
          {selected.record?.prompt && (<><h3>Prompt</h3><p>{selected.record.prompt}</p></>)}
          {selected.record?.openui && (<><h3>OpenUI target</h3><pre>{selected.record.openui}</pre></>)}
          <h3>Complete entry</h3>
          <pre>{JSON.stringify(selected, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export function TrainingDataGenerator({
  versions = [],
}: {
  versions?: string[];
}) {
  const caps = useCaps();
  const [version, setVersion] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const trainJob = jobDef(caps, "build_train_data");
  const outputVersion = version.trim() || `training-v${Math.max(1, versions.length)}`;

  async function launch() {
    if (!trainJob) return;
    setBusy(true);
    setError(null);
    try {
      const job = await postJSON<any>("/api/jobs", {
        job: "build_train_data",
        params: { version: outputVersion },
      });
      setVersion(outputVersion);
      setJobId(job.id);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="data-generator">
      <p className="launcher-summary">
        Build a ready-to-train dataset from all available examples. Variations,
        validation, and duplicate removal are handled automatically.
      </p>
      <label className="launcher-field data-generator-name">
        <span>Dataset name</span>
        <input
          type="text"
          value={version}
          placeholder={outputVersion}
          onChange={(e) => setVersion(e.target.value)}
        />
      </label>
      <ErrorNote error={error} />
      <button
        className="btn btn-primary"
        disabled={!caps.execution || !trainJob || busy || !outputVersion}
        onClick={launch}
        title={caps.execution ? "" : "Serve locally to enable generation"}
      >
        {busy ? "Starting…" : caps.execution ? "Create dataset" : "Read-only"}
      </button>
      <LogStream jobId={jobId} />
    </div>
  );
}

export function TrainingDataWorkspace({
  versions = [],
}: {
  versions?: string[];
}) {
  return <TrainingDataGenerator versions={versions} />;
}

export function Data({ navigate }: { navigate: (to: string) => void }) {
  const [version, setVersion] = useState<string | null>(() => {
    try {
      return new URLSearchParams(window.location.search).get("version");
    } catch {
      return null;
    }
  });
  const train = usePoll<any>(version ? `/api/data/train?version=${version}` : "/api/data/train", 3000);
  const test = usePoll<any>("/api/data/test", 0);
  const preference = usePoll<any>("/api/data/preference", 3000);

  const v = train.data?.version;
  const stats = train.data?.stats;

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Training Data</h1>
        <p className="page-sub">
          Review the examples the model learns from, or create a new ready-to-train dataset.
        </p>
      </div>

      <ErrorNote error={train.error} />

      {train.data?.versions?.length > 0 && (
        <div className="chip-row" style={{ marginBottom: "1rem" }}>
          {train.data.versions.map((vv: string) => (
            <span key={vv} className={`chip ${vv === v ? "active" : ""}`} onClick={() => setVersion(vv)}>{vv}</span>
          ))}
        </div>
      )}

      <Grid min="190px">
        <StatTile label="Records" value={train.data?.record_count ?? stats?.record_count ?? "—"} accent="moss" sub={v === "examples" ? "built-in examples" : train.data?.path ?? "no data"} />
        <StatTile label="Storage" value={train.data?.storage ?? "committed"} sub={train.data?.fingerprint?.slice(0, 12) ?? null} />
        <StatTile label="Build trace" value={train.data?.trace_id?.slice(0, 12) ?? "—"} sub={train.data?.trace_id ? "W3C trace ID" : null} />
        <StatTile label="Collected" value={stats?.collected_count ?? "—"} />
        <StatTile label="Quality rejected" value={stats?.quality_rejected ?? "—"} accent={stats?.quality_rejected ? "failed" : undefined} />
        <StatTile label="Synthesizer" value={stats?.synthesizer ?? "—"} />
        <StatTile label="Errors" value={stats?.error_count ?? "—"} accent={stats?.error_count ? "failed" : undefined} />
        <StatTile label="Profile" value={train.data?.profile ?? "—"} sub={train.data?.quality ? "curation profile" : null} />
        <StatTile label="Redundancy dropped" value={train.data?.quality?.redundancy_dropped ?? "—"} accent={train.data?.quality?.redundancy_dropped ? "ember" : undefined} />
        <StatTile label="Decontam flagged" value={train.data?.quality?.decontam_flagged ?? "—"} accent={train.data?.quality?.decontam_flagged ? "failed" : undefined} />
      </Grid>

      <Card title="Data quality report" right={<ProvenanceBadge provenance={train.data?.provenance} />}>
        <DataQualityCard version={v} usedByRuns={train.data?.used_by_runs ?? []} navigate={navigate} />
      </Card>

      <Card title="Training examples" right={<span className="hint">search · filter · inspect</span>}>
        <TrainingDataBrowser version={v} />
      </Card>

      <Card title="Rejected records (audit ledger)" right={<span className="hint">nothing is dropped silently</span>}>
        <RejectedRecordsBrowser version={v} />
      </Card>

      <Card title="Preference and decision data" right={<ProvenanceBadge provenance={preference.data?.provenance} />}>
        <DataTable
          columns={[
            { key: "dataset_id", label: "dataset" },
            { key: "kind", label: "kind" },
            { key: "records", label: "events", align: "right" },
            { key: "train", label: "train", align: "right" },
            { key: "held_out", label: "held-out", align: "right" },
            { key: "evidence", label: "evidence" },
            { key: "usage", label: "used by" },
            { key: "fingerprint", label: "fingerprint" },
          ]}
          rows={preference.data?.rows ?? []}
        />
      </Card>

      <Card title="Create a training dataset" right={<span className="hint">recommended settings included</span>}>
        <TrainingDataWorkspace versions={train.data?.versions ?? []} />
      </Card>

      <div className="two-col">
        <Card title="Corpus composition" right={<ProvenanceBadge provenance={train.data?.provenance} />}>
          {stats ? (
            <Bars
              data={[
                { label: "records", value: stats.record_count ?? 0 },
                { label: "collected", value: stats.collected_count ?? 0 },
                { label: "quality rejected", value: stats.quality_rejected ?? 0 },
                { label: "reserved-test rejected", value: stats.structure_reserved_rejected ?? 0 },
              ]}
            />
          ) : (
            <Bars data={Object.entries(train.data?.fixture_counts ?? {}).map(([k, val]) => ({ label: k, value: val as number }))} />
          )}
        </Card>

        <Card title="Eval suites" right={<ProvenanceBadge provenance={test.data?.provenance} />}>
          <Bars data={Object.entries(test.data?.suites ?? {}).map(([k, val]) => ({ label: k, value: val as number }))} />
          <p className="hint" style={{ marginTop: "0.5rem" }}>Ship claims require full <span className="mono">rico_held</span> (n≥1500).</p>
        </Card>
      </div>
    </div>
  );
}
