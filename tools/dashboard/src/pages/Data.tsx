import React, { useEffect, useState } from "react";
import { postJSON, usePoll } from "../api";
import { useCaps, jobDef } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  DataTable,
  Bars,
  JobLauncher,
  LogStream,
  ProvenanceBadge,
  Empty,
  ErrorNote,
} from "../components";

const PAGE_SIZE = 20;

export function TrainingDataBrowser({ version }: { version?: string }) {
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("");
  const [selected, setSelected] = useState<any>(null);
  const params = new URLSearchParams({
    limit: String(PAGE_SIZE),
    offset: String(page * PAGE_SIZE),
  });
  if (query) params.set("q", query);
  if (source) params.set("source", source);
  const recs = usePoll<any>(
    version ? `/api/data/train/${encodeURIComponent(version)}/records?${params}` : null,
    0,
  );

  useEffect(() => {
    setPage(0);
    setSelected(null);
    setQuery("");
    setSource("");
  }, [version]);

  useEffect(() => {
    setPage(0);
    setSelected(null);
  }, [query, source]);

  if (!version) return <Empty>No built corpus. Generate one below.</Empty>;
  const count = recs.data?.count ?? 0;
  const rows = recs.data?.records ?? [];
  const start = count ? page * PAGE_SIZE + 1 : 0;
  const end = Math.min((page + 1) * PAGE_SIZE, count);

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
            {(recs.data?.sources ?? []).map((item: string) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
        <span className="hint data-browser-count">Showing {start}–{end} of {count}</span>
      </div>
      <ErrorNote error={recs.error} />
      <DataTable
        columns={[
          { key: "id", label: "id" },
          { key: "source", label: "source" },
          { key: "task", label: "task" },
          { key: "prompt", label: "prompt" },
          { key: "view", label: "" },
        ]}
        rows={rows}
        render={{
          id: (r) => <span className="mono">{r.id}</span>,
          task: (r) => <span>{r.meta?.task ?? "generation"}</span>,
          prompt: (r) => <span className="data-prompt">{r.prompt}</span>,
          view: (r) => (
            <button className="btn btn-small" onClick={() => setSelected(r)}>
              View
            </button>
          ),
        }}
      />
      <div className="data-browser-pages">
        <button
          className="btn btn-small"
          disabled={page === 0 || recs.loading}
          onClick={() => setPage((value) => Math.max(0, value - 1))}
        >
          Previous
        </button>
        <span className="hint">Page {page + 1}</span>
        <button
          className="btn btn-small"
          disabled={end >= count || recs.loading}
          onClick={() => setPage((value) => value + 1)}
        >
          Next
        </button>
      </div>
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
        </div>
      )}
    </div>
  );
}

const RECIPES: Record<string, { label: string; help: string; synthesizer: string; namespace?: boolean; edits?: boolean; repairs?: number }> = {
  full: { label: "Full derivative mix", help: "Prompt, layout, namespace, edit, and repair variants.", synthesizer: "quality", namespace: true, edits: true, repairs: 1 },
  quality: { label: "Prompt + layout", help: "Balanced prompt paraphrases and structural layout changes.", synthesizer: "quality" },
  prompts: { label: "Prompt paraphrases", help: "New requests paired with the same OpenUI target.", synthesizer: "template" },
  layouts: { label: "Layout variations", help: "Stack direction and call-to-action structural variants.", synthesizer: "layout" },
  namespaces: { label: "Namespace variations", help: "Re-prefix placeholders without changing the layout intent.", synthesizer: "none", namespace: true },
  edits: { label: "Edit + repair tasks", help: "Generate edit trajectories and corrupted-program repair examples.", synthesizer: "none", edits: true, repairs: 2 },
  validate: { label: "Validate / copy only", help: "Revalidate, deduplicate, and version the selected roots without variants.", synthesizer: "none" },
};

export function TrainingDataGenerator({
  versions = [],
  selectedVersion,
}: {
  versions?: string[];
  selectedVersion?: string;
}) {
  const caps = useCaps();
  const [base, setBase] = useState("");
  const [recipe, setRecipe] = useState("quality");
  const [version, setVersion] = useState("v1");
  const [curriculum, setCurriculum] = useState(false);
  const [fuzzyDedup, setFuzzyDedup] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const trainJob = jobDef(caps, "build_train_data");

  useEffect(() => {
    if (selectedVersion) {
      setBase(`existing:${selectedVersion}`);
      setVersion(`${selectedVersion}-derived`);
      setRecipe("full");
    }
  }, [selectedVersion]);

  const activeBase = base || "fixture";
  const activeRecipe = RECIPES[recipe];
  const existingVersion = activeBase.startsWith("existing:")
    ? activeBase.slice("existing:".length)
    : null;
  const supportsEdits = !!existingVersion || ["programspec", "integrated", "all"].includes(activeBase);
  const incompatible = recipe === "edits" && !supportsEdits;

  async function launch() {
    if (!trainJob || incompatible) return;
    setBusy(true);
    setError(null);
    try {
      const params: Record<string, unknown> = {
        source: existingVersion ? "existing" : activeBase,
        version,
        synthesizer: activeRecipe.synthesizer,
        namespace_augment: !!activeRecipe.namespace,
        edit_derivatives: !!activeRecipe.edits,
        repairs_per_program: activeRecipe.repairs ?? 0,
        curriculum,
        fuzzy_dedup: fuzzyDedup,
      };
      if (existingVersion) params.base_version = existingVersion;
      const job = await postJSON<any>("/api/jobs", { job: "build_train_data", params });
      setJobId(job.id);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="data-generator">
      <div className="data-generator-fields">
        <label className="launcher-field">
          <span>Start from</span>
          <select value={activeBase} onChange={(e) => setBase(e.target.value)}>
            {versions.length > 0 && (
              <optgroup label="Existing generated corpus">
                {versions.map((item) => <option key={item} value={`existing:${item}`}>{item}</option>)}
              </optgroup>
            )}
            <optgroup label="New roots">
              <option value="fixture">Committed fixtures</option>
              <option value="rico">RICO screens</option>
              <option value="awwwards">Awwwards screens</option>
              <option value="both">Fixtures + RICO</option>
              <option value="rico+awwwards">RICO + Awwwards</option>
              <option value="programspec">Generated ProgramSpecs</option>
              <option value="language_contract">Language contract</option>
              <option value="deconstruct">Deconstructed web captures</option>
              <option value="render">Rendered examples</option>
              <option value="integrated">All program-first roots</option>
              <option value="all">All available roots</option>
            </optgroup>
          </select>
        </label>
        <label className="launcher-field">
          <span>Variation recipe</span>
          <select value={recipe} onChange={(e) => setRecipe(e.target.value)}>
            {Object.entries(RECIPES).map(([key, item]) => (
              <option key={key} value={key}>{item.label}</option>
            ))}
          </select>
        </label>
        <label className="launcher-field">
          <span>Output version</span>
          <input type="text" value={version} onChange={(e) => setVersion(e.target.value)} />
        </label>
      </div>
      <p className="launcher-summary">{activeRecipe.help}</p>
      {incompatible && <div className="error-note">Edit + repair tasks need an existing corpus or ProgramSpec roots.</div>}
      <div className="data-generator-options">
        <label><input type="checkbox" checked={curriculum} onChange={(e) => setCurriculum(e.target.checked)} /> Curriculum tags + stress cases</label>
        <label><input type="checkbox" checked={fuzzyDedup} onChange={(e) => setFuzzyDedup(e.target.checked)} /> Fuzzy deduplication</label>
      </div>
      <ErrorNote error={error} />
      <button
        className="btn btn-primary"
        disabled={!caps.execution || !trainJob || busy || incompatible || !version}
        onClick={launch}
        title={caps.execution ? "" : "Serve locally to enable generation"}
      >
        {busy ? "Starting…" : caps.execution ? "Generate training data" : "Read-only"}
      </button>
      <LogStream jobId={jobId} />
    </div>
  );
}

export function TrainingDataWorkspace({
  versions = [],
  selectedVersion,
}: {
  versions?: string[];
  selectedVersion?: string;
}) {
  const caps = useCaps();
  const [testJobId, setTestJobId] = useState<string | null>(null);
  return (
    <>
      <TrainingDataGenerator versions={versions} selectedVersion={selectedVersion} />
      <div style={{ marginTop: "1rem" }}>
        {jobDef(caps, "build_test_data") && (
          <JobLauncher
            jobDef={jobDef(caps, "build_test_data")!}
            execution={caps.execution}
            defaults={{ source: "both", version: "v1", train_version: "v1" }}
            onLaunched={setTestJobId}
          />
        )}
      </div>
      <LogStream jobId={testJobId} />
    </>
  );
}

export function Data({ navigate: _navigate }: { navigate: (to: string) => void }) {
  const [version, setVersion] = useState<string | null>(null);
  const train = usePoll<any>(version ? `/api/data/train?version=${version}` : "/api/data/train", 3000);
  const test = usePoll<any>("/api/data/test", 0);

  const v = train.data?.version;
  const stats = train.data?.stats;

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Training Data</h1>
        <p className="page-sub">
          Navigate, generate, and validate versioned corpora. Cold-start shows committed
          fixtures; generated corpora live under <span className="mono">outputs/train_data/</span>.
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
        <StatTile label="Records" value={train.data?.record_count ?? stats?.record_count ?? "—"} accent="moss" sub={v ? `train_data/${v}` : "no built corpus"} />
        <StatTile label="Collected" value={stats?.collected_count ?? "—"} />
        <StatTile label="Quality rejected" value={stats?.quality_rejected ?? "—"} accent={stats?.quality_rejected ? "failed" : undefined} />
        <StatTile label="Synthesizer" value={stats?.synthesizer ?? "—"} />
        <StatTile label="Errors" value={stats?.error_count ?? "—"} accent={stats?.error_count ? "failed" : undefined} />
      </Grid>

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

      <Card title="Browse records" right={<span className="hint">search · filter · inspect</span>}>
        <TrainingDataBrowser version={v} />
      </Card>

      <Card title="Generate training data" right={<span className="hint">new roots or derivatives</span>}>
        <TrainingDataWorkspace versions={train.data?.versions ?? []} selectedVersion={v} />
      </Card>
    </div>
  );
}
