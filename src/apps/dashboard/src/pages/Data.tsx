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

export function Data({ navigate: _navigate }: { navigate: (to: string) => void }) {
  const [version, setVersion] = useState<string | null>(null);
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
      </Grid>

      <Card title="Training examples" right={<span className="hint">search · filter · inspect</span>}>
        <TrainingDataBrowser version={v} />
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
