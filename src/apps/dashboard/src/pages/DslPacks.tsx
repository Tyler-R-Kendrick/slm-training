import React, { useState } from "react";
import { usePoll } from "../api";
import {
  Card,
  DataTable,
  ErrorNote,
  Grid,
  ProvenanceBadge,
  StatTile,
} from "../components";

const GRAMMAR_COLUMNS = [
  { key: "id", label: "Line", align: "right" as const },
  { key: "run_id", label: "Grammar" },
];

function compact(value: number, digits = 1) {
  if (!Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, {
    notation: value >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: digits,
  });
}

function duration(seconds: number) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 0.001) return `${compact(seconds * 1e6)} µs`;
  if (seconds < 1) return `${compact(seconds * 1e3)} ms`;
  return `${compact(seconds)} s`;
}

function parameterCount(value: unknown) {
  const match = String(value ?? "").match(/([0-9.]+)\s*([KMB])?/i);
  if (!match) return null;
  const scale = { K: 1e3, M: 1e6, B: 1e9 }[match[2]?.toUpperCase() as "K" | "M" | "B"] ?? 1;
  return Number(match[1]) * scale;
}

export function DslPackEstimator({ data }: { data: any }) {
  const target = data?.planning?.performance_target ?? {};
  const profiles = target?.profiles ?? [];
  const single = profiles.find((row: any) => row.id === "single_stream");
  const aggregate = profiles.find((row: any) => row.id === "aggregate");
  const machine = data?.planning?.machine ?? {};

  const perfRows = (data?.evidence?.perf ?? []).map((row: any) => {
    const observed = Number(row.tokens_per_sec);
    const targetFloorMin = Number(row.target_floor_tps_min);
    const targetFloorMax = Number(row.target_floor_tps_max);
    const gapMin = Number(row.target_gap_factor_min);
    const gapMax = Number(row.target_gap_factor_max);
    const gapDisplay =
      row.target_gap_relation === "above"
        ? `${compact(1 / gapMax)}–${compact(1 / gapMin)}× above`
        : row.target_gap_relation === "below"
          ? `${compact(gapMin)}–${compact(gapMax)}× below`
          : "straddles";
    return {
      ...row,
      observed_tps: Number.isFinite(observed) ? compact(observed, 3) : "—",
      target_floor:
        Number.isFinite(targetFloorMin) && Number.isFinite(targetFloorMax)
          ? targetFloorMin === targetFloorMax
            ? compact(targetFloorMin)
            : `${compact(targetFloorMin)}–${compact(targetFloorMax)}`
          : "unavailable",
      gap:
        Number.isFinite(gapMin) && Number.isFinite(gapMax)
          ? `${gapDisplay} target · ${row.target_gap_status}`
          : row.target_gap_status ?? "not comparable",
      p50: Number.isFinite(Number(row.latency_ms_p50))
        ? `${compact(Number(row.latency_ms_p50))} ms`
        : "—",
      quality: `parse ${compact(Number(row.parse_rate) * 100)}% · fidelity ${compact(
        Number(row.placeholder_fidelity) * 100,
      )}% · n=${row.n ?? "—"}`,
    };
  });
  const checkpointRows = (data?.evidence?.checkpoints ?? []).map((row: any) => {
    const parameters = parameterCount(row.parameters);
    const targetParameters = Number(target?.parameter_count);
    const ratio =
      parameters == null || !Number.isFinite(targetParameters)
        ? null
        : parameters / targetParameters;
    return {
      ...row,
      target_comparison:
        ratio == null
          ? "not comparable"
          : `${compact(ratio, 3)}× target model P`,
      evaluation: `${row.evaluation_status ?? "unknown"} · gate ${
        row.gate_pass == null ? "not attached" : row.gate_pass ? "pass" : "fail"
      }`,
    };
  });
  const profileRows = profiles.map((row: any) => ({
    ...row,
    target_range: `${compact(Number(row.floor_tps))}–${compact(
      Number(row.ceiling_tps),
    )} tok/s`,
    decode_range: `${duration(Number(row.ceiling_decode_seconds))}–${duration(
      Number(row.floor_decode_seconds),
    )}`,
    bottleneck: row.floor_bottleneck,
  }));
  const leverRows = (target?.levers ?? []).map((row: any) => ({
    ...row,
    current_display:
      typeof row.current === "number" ? compact(Number(row.current), 3) : row.current,
  }));
  const targetInputRows = [
    {
      label: "Whole-model parameters",
      value: target?.parameter_count ?? "—",
      unit: "parameters",
      source: `${target?.parameter_source?.role ?? "unavailable"} · ${
        target?.parameter_source?.run_id ?? "—"
      }`,
    },
    {
      label: "Structural forward envelope",
      value: (target?.forward_range ?? []).join("–") || "—",
      unit: "forwards/record",
      source: "AST work/span bound",
    },
    ...Object.entries(target?.grammar_inputs ?? {}).map(([key, value]) => ({
      label: key.replaceAll("_", " "),
      value,
      unit: "count",
      source: "grammar, tokenizer, or certified artifact",
    })),
    ...(data?.planning?.assumptions ?? []),
  ];
  const artifact = data?.evidence?.completion_artifact ?? {};
  const kernel = data?.evidence?.completion_kernel ?? {};
  const kernelTimings = (kernel.timings ?? []).map((row: any) => ({
    ...row,
    baseline: `${compact(Number(row.baseline_median_ms), 3)} ms`,
    candidate_range: `${compact(Number(row.candidate_min_ms), 3)}–${compact(
      Number(row.candidate_max_ms),
      3,
    )} ms`,
    candidate_median: `${compact(Number(row.candidate_median_ms), 3)} ms`,
    candidate_mad: `${compact(Number(row.candidate_mad_ms), 3)} ms`,
    speedup_display: `${compact(Number(row.speedup), 2)}×`,
    parity: row.equivalent == null ? "not stated" : row.equivalent ? "preserved" : "failed",
  }));
  const kernelWork = (kernel.work ?? []).map((row: any) => ({
    ...row,
    metric_display: String(row.metric).replaceAll("_", " "),
    baseline_display: row.baseline == null ? "—" : compact(Number(row.baseline)),
    value_display: compact(Number(row.value)),
  }));
  const kernelGates = (kernel.gates ?? []).map((row: any) => ({
    ...row,
    gate_display: String(row.gate).replaceAll("_", " "),
    result: row.pass ? "pass" : "fail",
  }));
  const warmTiming = kernelTimings.find((row: any) => row.id === "warm_card_open_comma");
  const compilerTiming = kernelTimings.find((row: any) => row.scope === "compiler phase");
  const batchForwards = kernelWork.find(
    (row: any) => row.scope === "compiler_batch_two" && row.metric === "forwards_count",
  );

  return (
    <>
      <Card
        title="Causal optimal throughput target"
        right={<ProvenanceBadge provenance="calculated target" />}
      >
        <Grid min="190px">
          <StatTile
            label="Single-stream target"
            value={
              single
                ? `${compact(Number(single.floor_tps))}–${compact(
                    Number(single.ceiling_tps),
                  )} tok/s`
                : "—"
            }
            sub="latency profile; no concurrency multiplier"
            accent="moss"
          />
          <StatTile
            label="Aggregate target"
            value={
              aggregate
                ? `${compact(Number(aggregate.floor_tps))}–${compact(
                    Number(aggregate.ceiling_tps),
                  )} tok/s`
                : "—"
            }
            sub={`${aggregate?.streams ?? "—"} declared independent streams`}
          />
          <StatTile
            label="Model work authority"
            value={
              target?.parameter_count
                ? compact(Number(target.parameter_count), 3)
                : "unavailable"
            }
            sub={`${target?.parameter_source?.role ?? "no checkpoint"} · ${
              target?.parameter_source?.run_id ?? "—"
            }`}
          />
          <StatTile
            label="Runtime host"
            value={`${machine.logical_cores ?? "—"} cores · ${machine.frequency_ghz ?? "—"} GHz`}
            sub={`${machine.architecture ?? "unknown"} · ${machine.memory_gb ?? "—"} GiB RAM`}
          />
        </Grid>
        <DataTable
          columns={[
            { key: "label", label: "Profile" },
            { key: "streams", label: "Streams", align: "right" },
            { key: "target_range", label: "Target range" },
            { key: "decode_range", label: "Decode wall / record" },
            { key: "bottleneck", label: "Floor bottleneck" },
          ]}
          rows={profileRows}
        />
        <DataTable
          columns={[
            { key: "label", label: "Declared input" },
            { key: "value", label: "Value", align: "right" },
            { key: "unit", label: "Unit" },
            { key: "source", label: "Authority / assumption" },
          ]}
          rows={targetInputRows}
        />
        <p className="hint">
          {target?.formula}. {target?.basis}
        </p>
        <p className="hint">
          Runtime detection: {machine.cpu_model ?? "unknown CPU"};{" "}
          {machine.frequency_source ?? "frequency source unavailable"}.
        </p>
      </Card>

      <div className="two-col">
        <Card title="Available TPS levers">
          <DataTable
            columns={[
              { key: "lever", label: "Lever" },
              { key: "current_display", label: "Current" },
              { key: "direction", label: "Needed direction" },
              { key: "requirement", label: "Constraint / action" },
              { key: "effect", label: "Causal effect" },
            ]}
            rows={leverRows}
            searchable
            searchPlaceholder="Filter TPS levers…"
          />
        </Card>
        <Card title="Why the previous target did not match">
          <DataTable
            columns={[
              { key: "code_or_input", label: "Dimension" },
              { key: "current", label: "Previous state" },
              { key: "required", label: "Required state" },
              { key: "impact", label: "Impact" },
            ]}
            rows={target?.mismatches ?? []}
          />
        </Card>
      </div>

      <Card
        title="Latest diagnostic performance vs modeled target"
        right={<ProvenanceBadge provenance={data?.evidence?.perf_provenance} />}
      >
        <DataTable
          columns={[
            { key: "id", label: "Evidence" },
            { key: "run_id", label: "Run" },
            { key: "observed_tps", label: "Observed TPS", align: "right" },
            { key: "target_floor", label: "Target floor / range", align: "right" },
            { key: "gap", label: "Gap" },
            { key: "p50", label: "p50" },
            { key: "quality", label: "Quality / n" },
            { key: "evidence_class", label: "Claim class" },
          ]}
          rows={perfRows}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          {data?.evidence?.basis} These low-n perf rows are diagnostic controls, not ship
          or generalization evidence.
        </p>
      </Card>

      <Card
        title="Latest completion-kernel calculations from main"
        right={
          <>
            <span className="hint mono">{kernel.path ?? "—"}</span>
            <ProvenanceBadge provenance={kernel.provenance} />
          </>
        }
      >
        <Grid min="190px">
          <StatTile
            label="Warm domain speedup"
            value={warmTiming?.speedup_display ?? "—"}
            sub="persistent completion session"
            accent="moss"
          />
          <StatTile
            label="Compiler-phase speedup"
            value={compilerTiming?.speedup_display ?? "—"}
            sub="V1 compiler time ÷ packed compiler time"
            accent="moss"
          />
          <StatTile
            label="Batch forwards"
            value={
              batchForwards
                ? `${batchForwards.baseline_display} → ${batchForwards.value_display}`
                : "—"
            }
            sub="sequential control → compact ambiguous rows"
          />
          <StatTile
            label="Fixture gates"
            value={`${kernelGates.filter((row: any) => row.pass).length}/${kernelGates.length}`}
            sub={`${kernel.claim_class ?? "unknown claim class"} · n=${
              kernel.repetitions ?? "—"
            } repetitions`}
            accent={
              kernelGates.length > 0 && kernelGates.every((row: any) => row.pass)
                ? "moss"
                : "ember"
            }
          />
        </Grid>
        <DataTable
          columns={[
            { key: "scope", label: "Calculation" },
            { key: "id", label: "Fixture" },
            { key: "mode", label: "Mode" },
            { key: "candidate_count", label: "Candidates", align: "right" },
            { key: "baseline", label: "V1/control median", align: "right" },
            { key: "candidate_range", label: "Packed min–max", align: "right" },
            { key: "candidate_median", label: "Packed median", align: "right" },
            { key: "candidate_mad", label: "Packed MAD", align: "right" },
            { key: "speedup_display", label: "Speedup", align: "right" },
            { key: "parity", label: "Authority/parity" },
          ]}
          rows={kernelTimings}
        />
        <div className="two-col" style={{ marginTop: "0.8rem" }}>
          <div>
            <p className="hint">Packed-kernel work counters</p>
            <DataTable
              columns={[
                { key: "scope", label: "Fixture" },
                { key: "metric_display", label: "Counter" },
                { key: "baseline_display", label: "V1/control", align: "right" },
                { key: "value_display", label: "Packed", align: "right" },
              ]}
              rows={kernelWork}
              searchable
              searchPlaceholder="Filter kernel counters…"
            />
          </div>
          <div>
            <p className="hint">Committed fixture gates</p>
            <DataTable
              columns={[
                { key: "gate_display", label: "Gate" },
                { key: "result", label: "Result" },
              ]}
              rows={kernelGates}
            />
          </div>
        </div>
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          These are the committed packed completion-kernel timing summaries, work
          counters, parity checks, and calculated speedups added on current main. They
          are fixture/scratch evidence and remain independent of the optimal target
          formula above.
        </p>
      </Card>

      <Card title="Current checkpoints and model-work authority">
        <DataTable
          columns={[
            { key: "role", label: "Reference" },
            { key: "run_id", label: "Run / checkpoint" },
            { key: "parameters", label: "Parameters" },
            { key: "target_comparison", label: "Relative to target" },
            { key: "throughput", label: "Throughput" },
            { key: "evaluation", label: "Evaluation / gate" },
            { key: "provenance", label: "Evidence" },
          ]}
          rows={checkpointRows}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          The selected whole-model parameter count is a cost input to the target. The
          separate grammar-memory estimate above remains an information-capacity diagnostic,
          never a substitute for model work.
        </p>
        {(checkpointRows ?? []).map((row: any) => (
          <p className="hint" key={`${row.role}-${row.run_id}`}>
            {row.role}: {row.resource_basis}
          </p>
        ))}
      </Card>

      <Card title="Reusable certified completion artifact">
        <Grid min="190px">
          <StatTile
            label="Artifact status"
            value={String(artifact.status ?? "missing").replaceAll("_", " ")}
            sub={artifact.schema ?? "no manifest"}
            accent={artifact.status === "hash_verified" ? "moss" : "ember"}
          />
          <StatTile
            label="Static LALR graph"
            value={`${artifact.lalr_state_count ?? "—"} states`}
            sub={`${artifact.lalr_edge_count ?? "—"} certified control edges`}
          />
          <StatTile
            label="Direct token rows"
            value={compact(Number(artifact.direct_token_count))}
            sub="tokenizer → grammar-terminal authority"
          />
          <StatTile
            label="Dynamic overlay"
            value={`${artifact.request_dependent_authority_excluded?.length ?? 0} exclusions`}
            sub="rebuilt per request; never baked into the artifact"
          />
        </Grid>
        <p className="hint mono">{artifact.path ?? "—"}</p>
        <p className="hint">{artifact.runtime_policy}</p>
        <p className="hint">
          Request-local authority:{" "}
          {(artifact.request_dependent_authority_excluded ?? []).join(", ") || "—"}.
        </p>
      </Card>

      <Card
        title="Latest canonical smoke-suite quality"
        right={<ProvenanceBadge provenance={data?.evidence?.smoke_provenance} />}
      >
        {(data?.evidence?.smoke ?? []).length ? (
          <DataTable
            columns={[
              { key: "id", label: "Evidence" },
              { key: "run_id", label: "Run" },
              { key: "n", label: "n", align: "right" },
              { key: "meaningful_program_rate", label: "Meaningful" },
              { key: "placeholder_fidelity", label: "Fidelity" },
              { key: "reward_score", label: "Reward" },
              { key: "pass", label: "Pass" },
              { key: "evidence_class", label: "Claim class" },
            ]}
            rows={data.evidence.smoke}
          />
        ) : (
          <p className="hint">
            No canonical quality scoreboard row currently carries a smoke suite. The
            diagnostic n=1 performance evidence above is shown without relabeling it as a
            quality smoke result.
          </p>
        )}
      </Card>
    </>
  );
}

export function DslPacks() {
  const [pack, setPack] = useState("openui");
  const response = usePoll<any>(`/api/dsl-packs?pack=${encodeURIComponent(pack)}`, 0);
  const data = response.data;

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">DSL Packs</h1>
        <p className="page-sub">
          Inspect each pack&apos;s grammar and constraints, configurable training parameters,
          and research-backed targets computable from its structural patterns.
        </p>
      </div>

      <ErrorNote error={response.error} />

      <div className="chip-row" style={{ marginBottom: "1.15rem" }}>
        {(data?.packs ?? [pack]).map((id: string) => (
          <button
            key={id}
            className={`chip ${id === pack ? "active" : ""}`}
            aria-pressed={id === pack}
            onClick={() => setPack(id)}
          >
            {id}
          </button>
        ))}
      </div>

      <Grid min="190px">
        {(data?.tiles ?? []).map((tile: any) => (
          <StatTile key={tile.label} {...tile} />
        ))}
      </Grid>

      <Card
        title="Grammar source"
        right={
          <>
            <span className="hint mono">{data?.grammar?.path ?? "—"}</span>
            <ProvenanceBadge provenance={data?.provenance} />
          </>
        }
      >
        <DataTable columns={GRAMMAR_COLUMNS} rows={data?.grammar?.rows ?? []} />
      </Card>

      <div className="two-col">
        <Card title="Additional constraints">
          <DataTable
            columns={[
              { key: "constraint", label: "Constraint" },
              { key: "value", label: "Value" },
              { key: "authority", label: "Authority" },
            ]}
            rows={data?.constraints ?? []}
          />
        </Card>

        <Card title="Template syntax">
          <DataTable
            columns={[
              { key: "kind", label: "Kind" },
              { key: "syntax", label: "Syntax" },
              { key: "authority", label: "Authority" },
            ]}
            rows={data?.templates ?? []}
          />
        </Card>
      </div>

      <Card title="Pack facts and configurable parameters">
        <DataTable
          columns={[
            { key: "scope", label: "Role" },
            { key: "boundary", label: "Input" },
            { key: "minimum", label: "Min", align: "right" },
            { key: "maximum", label: "Max", align: "right" },
            { key: "authority", label: "Authority" },
          ]}
          rows={data?.boundaries ?? []}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          Grammar limits describe the language. Training and experiment-policy rows are
          configurable inputs; computed targets appear only below.
        </p>
      </Card>

      <Card title="Research-backed optimal targets">
        <Grid min="190px">
          {(data?.calculated?.tiles ?? []).map((tile: any) => (
            <StatTile key={tile.label} {...tile} />
          ))}
        </Grid>
        <DataTable
          columns={[
            { key: "metric", label: "Metric" },
            { key: "minimum", label: "Min", align: "right" },
            { key: "maximum", label: "Max", align: "right" },
            { key: "unit", label: "Unit" },
            { key: "formula", label: "Formula" },
            { key: "inputs", label: "Derived from" },
            { key: "cap_use", label: "Cap use" },
          ]}
          rows={data?.calculated?.rows ?? []}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          {data?.calculated?.basis ?? "Waiting for pack parameters…"}
        </p>
      </Card>

      <DslPackEstimator data={data} />

      <Card title="Not identifiable from pack parameters">
        <DataTable
          columns={[
            { key: "metric", label: "Requested target" },
            { key: "reason", label: "Why it cannot be estimated" },
            { key: "required", label: "Required inputs" },
            { key: "formula", label: "Valid relation" },
          ]}
          rows={data?.calculated?.unavailable ?? []}
        />
      </Card>

      <Card title="Research basis">
        <DataTable
          columns={[
            { key: "method", label: "Method" },
            { key: "reference", label: "Primary reference" },
            { key: "supports", label: "Used for" },
          ]}
          rows={data?.calculated?.references ?? []}
        />
      </Card>
    </div>
  );
}
