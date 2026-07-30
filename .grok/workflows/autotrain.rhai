// autotrain — continuous model + canonical harness improvement orchestration
//
// Host-owned workflow over the autotrain skill / slm CLI / OpenWiki.
// Phase order + exact commands are script-owned; agents execute and verify,
// they do not invent alternate harnesses or shadow roots.
//
// Args (object):
//   recipe          smoke|data|sft-eval|ship|matrix|custom   (default smoke)
//   run_id          durable id (required unless dry_run)
//   train_version   default: smoke→wf_v0 else v1
//   eval_version    default: same as train_version
//   train_dir       override (default outputs/data/train/<train_version>)
//   test_dir        override (default outputs/data/eval/<eval_version>)
//   steps           default: smoke 8, matrix 40, ship 80, else 20
//   device          default cpu
//   context_backend scratch|hf  (default scratch; ship→hf)
//   model           default twotower
//   output_tokenizer choice|lexer  (default choice; never free-form compositional for smoke)
//   ship_gates      bool (ship recipe forces true)
//   matrix          quality matrix version (default v6)
//   only            matrix --only (default E53)
//   phases          custom phase id array
//   dry_run         plan only — no execute agents after inspect
//   allow_paid      required for hf-jobs/remote/bucket write
//   continuous      default true; false/once runs one finite pass
//   once            alias for continuous=false
//   skip_completed  default true (durable state under outputs/autoresearch/<run_id>/)
//   seed            default 0
//   force_phases    never skip these phase ids
//   cli             command prefix (default:
//                   "uv run --extra torch python -m scripts.slm")
//                   data-only may use "uv run python -m scripts.slm"
//                   or plain "slm" when the console script is installed
//   eval_limit      optional diagnostic cap for smoke eval (default 3 on smoke)
//   source          data source override (smoke default fixture)
//   publish_data    if true, allow build-train --publish into resources (default false)
//
// Phase ids: data_train data_test quality sft distill preference rl eval
//            matrix closeout verify
//
// Docs agents must read (in order):
//   AGENTS.md                                       (canonical for every agent)
//   docs/design/decode-invariants.md                (goal law)
//   docs/openwiki/quickstart.md
//   docs/openwiki/workflows/train-eval-docs.md
//   docs/openwiki/operations/checkpoints-and-agents.md
//   .agents/skills/autotrain/SKILL.md
//   .agents/skills/autotrain/references/contracts.md
//   .agents/skills/autotrain/references/<phase>.md  (via `slm guide <slug>`)
//
// Contracts: AGENTS.md is canonical and outranks tool defaults on process
// conflicts. Non-negotiable architecture invariants live in
// docs/design/decode-invariants.md -- never weaken them; a rejected experiment
// closes an approach, never a goal. Iron law docs (skill:
// documenting-experiment-results), model card (docs/MODEL_CARD.md + README
// summary), fixture≠ship, RL fail-closed, improve-openui-harnesses only after a
// frozen reproduced signal, Lean certificate replay / band repair via
// improve-lean-optimums, fetch+merge origin/main between runs, three result
// matrices via `slm autoresearch status --loop-id <run_id> --matrix --last 5`, hard run cap
// (slm_training.levers.MAX_RUN_MINUTES), no paid without allow_paid.

let meta = #{
    name: "autotrain",
    description: "Deterministic autotrain orchestration: recipe→exact slm commands, parallel data legs, durable state, adversarial verify",
    when_to_use: "Operate OpenUI train/eval/matrix via slm harnesses with durable, observable multi-agent orchestration instead of a free-form agent",
    phases: [
        #{ title: "Plan", detail: "expand recipe, exact command plan, durable skips" },
        #{ title: "Data", detail: "parallel train/test corpus via slm data" },
        #{ title: "Quality", detail: "synthesis-feedback gate on quality_report" },
        #{ title: "Train", detail: "SFT/distill/preference/RL via slm" },
        #{ title: "Eval", detail: "slm eval model + optional ship gates" },
        #{ title: "Matrix", detail: "slm experiments * matrix" },
        #{ title: "Closeout", detail: "docs/design + MODEL_CARD (iron law)" },
        #{ title: "Verify", detail: "adversarial evidence panel" },
        #{ title: "Report", detail: "scratch report + durable state" },
    ],
};

// ── schemas ──────────────────────────────────────────────────────────────

let inspect_schema = #{
    "type": "object",
    "required": ["completed_phases", "blockers", "prereqs_ok"],
    "properties": #{
        "completed_phases": #{
            "type": "array", "maxItems": 16,
            "items": #{ "type": "string" },
        },
        "blockers": #{
            "type": "array", "maxItems": 8,
            "items": #{ "type": "string" },
        },
        "prereqs_ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let phase_result_schema = #{
    "type": "object",
    "required": ["phase", "status", "summary", "commands_run"],
    "properties": #{
        "phase": #{ "type": "string" },
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "commands_run": #{
            "type": "array", "maxItems": 12,
            "items": #{ "type": "string" },
        },
        "artifacts": #{
            "type": "array", "maxItems": 24,
            "items": #{ "type": "string" },
        },
        "docs_updated": #{
            "type": "array", "maxItems": 12,
            "items": #{ "type": "string" },
        },
        "exit_codes": #{
            "type": "array", "maxItems": 12,
            "items": #{ "type": "integer" },
        },
        "error": #{ "type": "string" },
    },
};

let claim_schema = #{
    "type": "object",
    "required": ["claims"],
    "properties": #{
        "claims": #{
            "type": "array", "maxItems": 6,
            "items": #{
                "type": "object",
                "required": ["id", "claim", "evidence_path"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "claim": #{ "type": "string" },
                    "evidence_path": #{ "type": "string" },
                },
            },
        },
    },
};

let verdict_schema = #{
    "type": "object",
    "required": ["claim_id", "real", "reason", "evidence"],
    "properties": #{
        "claim_id": #{ "type": "string" },
        "real": #{ "type": "boolean" },
        "reason": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

// ── helpers ──────────────────────────────────────────────────────────────

fn str_or(v, fallback) {
    if type_of(v) == "string" {
        let t = v;
        t.trim();
        if t != "" { return t; }
    }
    fallback
}

fn bool_or(v, fallback) {
    if type_of(v) == "bool" { v } else { fallback }
}

fn int_or(v, fallback) {
    let t = type_of(v);
    if t == "i64" || t == "i32" || t == "u64" || t == "u32" || t == "f64" || t == "f32" {
        v
    } else {
        fallback
    }
}

fn contains_str(arr, needle) {
    if type_of(arr) != "array" { return false; }
    for x in arr {
        if x == needle { return true; }
    }
    false
}

fn is_known_phase(p) {
    p == "data_train" || p == "data_test" || p == "quality" || p == "sft"
        || p == "distill" || p == "preference" || p == "rl" || p == "eval"
        || p == "matrix" || p == "closeout" || p == "verify"
}

fn filter_known_phases(raw) {
    let out = [];
    if type_of(raw) != "array" { return out; }
    for p in raw {
        if type_of(p) == "string" && is_known_phase(p) && !contains_str(out, p) {
            out.push(p);
        }
    }
    out
}

fn recipe_phases(recipe) {
    if recipe == "data" {
        return ["data_train", "data_test", "quality", "closeout", "verify"];
    }
    if recipe == "sft-eval" {
        return ["sft", "eval", "closeout", "verify"];
    }
    if recipe == "ship" {
        return ["data_train", "data_test", "quality", "sft", "eval", "closeout", "verify"];
    }
    if recipe == "matrix" {
        return ["matrix", "closeout", "verify"];
    }
    if recipe == "custom" {
        return [];
    }
    // smoke — fixture wiring under hard run cap
    ["data_train", "data_test", "quality", "sft", "eval", "closeout", "verify"]
}

fn join_list(arr) {
    if type_of(arr) != "array" || arr.len() == 0 { return "(none)"; }
    let s = "";
    let i = 0;
    for p in arr {
        if i > 0 { s += ", "; }
        s += p.to_string();
        i += 1;
    }
    s
}

fn join_cmds(arr) {
    if type_of(arr) != "array" || arr.len() == 0 { return "(none)"; }
    let s = "";
    let i = 0;
    for p in arr {
        if i > 0 { s += "\n"; }
        s += p.to_string();
        i += 1;
    }
    s
}

fn push_event(events, phase_name, status, summary) {
    let out = events;
    out.push(#{ phase: phase_name, status: status, summary: summary });
    out
}

fn contracts_text() {
    let s = "";
    s += "CONTRACTS (mandatory):\n";
    s += "- Iron law: train/eval/bench/matrix → docs/design JSON + matching markdown (skill documenting-experiment-results).\n";
    s += "- Model card: checkpoint create/sync/promote → docs/MODEL_CARD.md + README Model card (summary).\n";
    s += "- Honesty: fixture/scratch/smoke is wiring only; readiness needs --ship-gates on full multi-suite scoreboard (honest-ship-eval).\n";
    s += "- RL fail-closed: approved RLReadinessReport required (slm autoresearch validate-rl); no override.\n";
    s += "- No shadow paths: only slm / python -m scripts.* and outputs/{data,runs,autoresearch}/ + docs/design/.\n";
    s += "- Continuous source: commit cycle, fetch and merge origin/main, resolve conflicts, and fix every check before the next run.\n";
    s += "- Lean optimums: replay v2 certificates with improve-lean-optimums; theorem miss stops the campaign, assumption miss blocks promotion and opens all five lanes.\n";
    s += "- Hard run cap: slm_training.levers.MAX_RUN_MINUTES (currently 3m). Timed-out runs are not evidence.\n";
    s += "- Approvals: no paid GPU/remote/HF write unless allow_paid=true.\n";
    s += "- OpenWiki (read, do not hand-edit generated pages): docs/openwiki/quickstart.md, docs/openwiki/workflows/train-eval-docs.md, docs/openwiki/operations/checkpoints-and-agents.md.\n";
    s
}

fn openwiki_block() {
    let s = "";
    s += "OPENWIKI / SKILL MAP (read with tools before acting):\n";
    s += "1. docs/openwiki/quickstart.md\n";
    s += "2. docs/openwiki/workflows/train-eval-docs.md\n";
    s += "3. docs/openwiki/operations/checkpoints-and-agents.md\n";
    s += "4. .agents/skills/autotrain/SKILL.md\n";
    s += "5. .agents/skills/autotrain/references/contracts.md\n";
    s += "6. Phase guide: run `<cli> guide <slug>` or read .agents/skills/autotrain/references/<slug>.md\n";
    s += "   slugs: train-data test-data sft eval distill preference rl experiments checkpoints contracts\n";
    s
}

// Deterministic command builders — agents must run these, not invent alternatives.
fn cmds_data_train(cli, source, train_version, seed, allow_publish) {
    let c = [];
    let cmd = cli;
    cmd += " data build-train --source ";
    cmd += source;
    cmd += " --version ";
    cmd += train_version;
    cmd += " --synthesizer quality --programspec-seed ";
    cmd += seed.to_string();
    // Smoke/scratch builds stay under outputs/ — never auto-publish into
    // src/slm_training/resources/data unless explicitly allowed.
    if !allow_publish {
        cmd += " --no-publish";
    }
    c.push(cmd);
    c
}

fn cmds_data_test(cli, source, eval_version, train_dir, seed) {
    let c = [];
    let cmd = cli;
    cmd += " data build-test --source ";
    cmd += source;
    cmd += " --version ";
    cmd += eval_version;
    cmd += " --train-manifest ";
    cmd += train_dir;
    cmd += "/manifest.json";
    c.push(cmd);
    // Canonical root listing (data store) — verify wiring, not a second corpus tree
    c.push(cli + " data store list");
    c
}

fn cmds_sft(cli, train_dir, model_name, context_backend, steps, run_id, device, seed, allow_paid, output_tokenizer) {
    let c = [];
    let cmd = cli;
    cmd += " sft train --train-dir ";
    cmd += train_dir;
    cmd += " --model ";
    cmd += model_name;
    cmd += " --context-backend ";
    cmd += context_backend;
    cmd += " --output-tokenizer ";
    cmd += output_tokenizer;
    cmd += " --steps ";
    cmd += steps.to_string();
    cmd += " --run-id ";
    cmd += run_id;
    cmd += " --device ";
    cmd += device;
    cmd += " --seed ";
    cmd += seed.to_string();
    cmd += " --fast-train";
    if context_backend != "hf" || !allow_paid {
        cmd += " --no-sync-checkpoints";
    }
    c.push(cmd);
    c
}

fn cmds_eval(cli, test_dir, model_name, run_id, device, ship_gates, eval_limit) {
    let c = [];
    let cmd = cli;
    cmd += " eval model --test-dir ";
    cmd += test_dir;
    cmd += " --model ";
    cmd += model_name;
    cmd += " --run-id ";
    cmd += run_id;
    cmd += " --device ";
    cmd += device;
    if ship_gates {
        cmd += " --ship-gates";
    } else if eval_limit > 0 {
        cmd += " --eval-limit ";
        cmd += eval_limit.to_string();
        cmd += " --suites smoke";
    }
    c.push(cmd);
    c
}

fn cmds_matrix(cli, matrix_ver, only_id, steps, device, context_backend) {
    let c = [];
    let cmd = cli;
    cmd += " experiments quality-matrix --matrix ";
    cmd += matrix_ver;
    cmd += " --only ";
    cmd += only_id;
    cmd += " --steps ";
    cmd += steps.to_string();
    cmd += " --device ";
    cmd += device;
    cmd += " --context-backend ";
    cmd += context_backend;
    if context_backend == "scratch" {
        cmd += " --scratch-control --no-design-md-context";
    }
    c.push(cmd);
    c
}

fn cmds_distill(cli, run_id) {
    let c = [];
    c.push(cli + " distill collect --help");
    // Prefer guide first; real collect needs campaign-specific paths — agent must
    // read `slm guide distill` and only run collect/self with config paths if present.
    c.push("# AFTER reading guide: slm distill collect|self|resume-climb with campaign args (no shadow roots)");
    c
}

fn cmds_preference(cli, run_dir) {
    let c = [];
    c.push(cli + " preference --help");
    c.push("# AFTER guide: slm preference build-pairs ... then slm preference train --from-checkpoint " + run_dir + "/checkpoints/last.pt ...");
    c
}

fn cmds_rl(cli, run_dir, train_dir, workflow_dir) {
    let c = [];
    // Fail-closed: only if approved report exists
    let report = workflow_dir + "/rl_readiness_report.json";
    c.push("test -f " + report + " || test -f outputs/autoresearch/*/rl_readiness_report.json");
    c.push(cli + " rl train --checkpoint " + run_dir + "/checkpoints/last.pt --train-records " + train_dir + " --rl-readiness-report " + report + " --steps 8 --group-size 4");
    c
}

fn expected_artifacts(phase_id, train_dir, test_dir, run_dir, workflow_dir, state_path) {
    let a = [];
    if phase_id == "data_train" {
        a.push(train_dir + "/manifest.json");
        a.push(train_dir + "/quality_report.json");
        a.push(train_dir + "/rejected.jsonl");
        a.push(train_dir + "/synthesis_feedback.json");
    } else if phase_id == "data_test" {
        a.push(test_dir);
    } else if phase_id == "quality" {
        a.push(train_dir + "/quality_report.json");
        a.push(train_dir + "/synthesis_feedback.json");
        a.push(workflow_dir + "/quality_gate.json");
    } else if phase_id == "sft" {
        // train_model writes train_summary.json at the run root (and may use checkpoints/)
        a.push(run_dir + "/train_summary.json");
        a.push(run_dir + "/checkpoints/last.pt");
    } else if phase_id == "eval" {
        a.push(run_dir + "/scoreboard.json");
        a.push(run_dir + "/gates.json");
        a.push(run_dir + "/agentv");
    } else if phase_id == "matrix" {
        a.push("docs/design/quality-matrix-results.json");
        a.push("docs/design/quality-experiment-matrix.md");
    } else if phase_id == "closeout" {
        a.push(state_path);
    } else if phase_id == "verify" {
        a.push(state_path);
    }
    a
}

fn guide_slug(phase_id) {
    if phase_id == "data_train" { return "train-data"; }
    if phase_id == "data_test" { return "test-data"; }
    if phase_id == "quality" { return "train-data"; }
    if phase_id == "sft" { return "sft"; }
    if phase_id == "distill" { return "distill"; }
    if phase_id == "preference" { return "preference"; }
    if phase_id == "rl" { return "rl"; }
    if phase_id == "eval" { return "eval"; }
    if phase_id == "matrix" { return "experiments"; }
    if phase_id == "closeout" { return "contracts"; }
    if phase_id == "verify" { return "contracts"; }
    "contracts"
}

fn phase_commands(phase_id, cfg) {
    let cli = cfg.cli;
    if phase_id == "data_train" {
        return cmds_data_train(cli, cfg.source, cfg.train_version, cfg.seed, cfg.publish_data);
    }
    if phase_id == "data_test" {
        return cmds_data_test(cli, cfg.source, cfg.eval_version, cfg.train_dir, cfg.seed);
    }
    if phase_id == "quality" {
        // no heavy CLI — read feedback artifacts; optional store verify
        let c = [];
        c.push(cli + " data store verify train " + cfg.train_version);
        c.push("# READ " + cfg.train_dir + "/quality_report.json rejected.jsonl synthesis_feedback.json");
        c.push("# WRITE " + cfg.workflow_dir + "/quality_gate.json summarizing status (never weaken gates)");
        return c;
    }
    if phase_id == "sft" {
        return cmds_sft(cli, cfg.train_dir, cfg.model, cfg.context_backend, cfg.steps, cfg.run_id, cfg.device, cfg.seed, cfg.allow_paid, cfg.output_tokenizer);
    }
    if phase_id == "distill" {
        return cmds_distill(cli, cfg.run_id);
    }
    if phase_id == "preference" {
        return cmds_preference(cli, cfg.run_dir);
    }
    if phase_id == "rl" {
        return cmds_rl(cli, cfg.run_dir, cfg.train_dir, cfg.workflow_dir);
    }
    if phase_id == "eval" {
        return cmds_eval(cli, cfg.test_dir, cfg.model, cfg.run_id, cfg.device, cfg.ship_gates, cfg.eval_limit);
    }
    if phase_id == "matrix" {
        return cmds_matrix(cli, cfg.matrix, cfg.only, cfg.steps, cfg.device, cfg.context_backend);
    }
    if phase_id == "closeout" {
        let c = [];
        c.push("# READ skill .agents/skills/documenting-experiment-results/SKILL.md");
        c.push("# UPDATE docs/design JSON+markdown for this run; MODEL_CARD+README if checkpoint");
        c.push("# WRITE/MERGE " + cfg.state_path);
        return c;
    }
    if phase_id == "verify" {
        let c = [];
        c.push("# harvest claims from artifacts only — no new training");
        return c;
    }
    []
}

fn result_status(r) {
    if r == () || !r.success || r.output == () { return "failed"; }
    let st = str_or(r.output.status, "failed");
    if st == "ok" || st == "skipped" || st == "failed" { st } else { "failed" }
}

fn result_summary(r, phase_id) {
    if r == () { return phase_id + " agent slot empty"; }
    if !r.success { return phase_id + " agent success=false"; }
    if r.output == () { return phase_id + " empty output"; }
    str_or(r.output.summary, phase_id + " completed")
}

fn exec_prompt(phase_id, cfg_json, commands, artifacts, contracts, wiki) {
    let slug = guide_slug(phase_id);
    let p = "";
    p += "You EXECUTE autotrain phase `";
    p += phase_id;
    p += "` using the repo's canonical harnesses only.\n";
    p += wiki;
    p += contracts;
    p += "\nCONFIG JSON (authoritative):\n";
    p += cfg_json;
    p += "\n\nREAD FIRST (tools):\n";
    p += "- Run: the cli from config + ` guide ";
    p += slug;
    p += "`  OR read .agents/skills/autotrain/references/";
    p += slug;
    p += ".md\n";
    p += "- Skim OpenWiki paths listed above.\n";
    p += "\nEXACT COMMANDS TO RUN (in order). Do not substitute alternate trainers/evaluators.\n";
    p += "Lines starting with # are instructions, not shell.\n";
    p += join_cmds(commands);
    p += "\n\nEXPECTED ARTIFACTS TO VERIFY AFTERWARD:\n";
    p += join_cmds(artifacts);
    p += "\n\nRULES:\n";
    p += "1. Create workflow_dir if missing. Merge phase status into state_path JSON after work.\n";
    p += "2. Idempotent: if expected artifacts already valid, verify them and status=skipped (do not rebuild).\n";
    p += "3. Use run_terminal_command from repo root. Prefer config.cli prefix. If cli fails, fall back once to `uv run python -m scripts.<module>` matching `slm list` mapping — still no shadow scripts.\n";
    p += "4. Obey hard run cap. Prefer smaller fixture/smoke settings already in the commands.\n";
    p += "5. For quality: follow synthesis-feedback skill — fix nothing in gates; summarize feedback; write quality_gate.json.\n";
    p += "6. For closeout: documenting-experiment-results; never claim ship from fixture.\n";
    p += "7. For rl: if readiness report missing → status=failed (do not train).\n";
    p += "8. Prereqs (once per machine/checkout):\n";
    p += "   - DSL bridge: `cd src/apps/openui_bridge && npm ci` before data builds.\n";
    p += "   - AgentV SDK: root `npm ci` before any `slm eval` (docs/openwiki + contracts).\n";
    p += "   - Torch: use cli with `uv run --extra torch` (or install the torch extra).\n";
    p += "9. Return structured output: status ok|failed|skipped, commands_run actually executed, artifacts verified, summary, exit_codes.\n";
    p
}

// ── args ─────────────────────────────────────────────────────────────────

let a = args;
if a == () {
    pause("verification", "Pass args object. Example: {\"recipe\":\"smoke\",\"run_id\":\"autotrain_smoke_1\",\"dry_run\":true}");
}

let recipe = str_or(a.recipe, "smoke");
let dry_run = bool_or(a.dry_run, false);
let run_id = str_or(a.run_id, "");
if run_id == "" {
    if dry_run {
        run_id = "autotrain_dry_run";
    } else {
        pause("verification", "args.run_id required for durable state under outputs/autoresearch/<run_id>/");
    }
}

let train_version_default = "v1";
if recipe == "smoke" { train_version_default = "wf_v0"; }
let train_version = str_or(a.train_version, train_version_default);
let eval_version = str_or(a.eval_version, train_version);
let train_dir = str_or(a.train_dir, "outputs/data/train/" + train_version);
let test_dir = str_or(a.test_dir, "outputs/data/eval/" + eval_version);

let steps_default = 20;
if recipe == "smoke" { steps_default = 8; }
if recipe == "matrix" { steps_default = 40; }
if recipe == "ship" { steps_default = 80; }
let steps = int_or(a.steps, steps_default);

let device = str_or(a.device, "cpu");
let context_backend = str_or(a.context_backend, "scratch");
if recipe == "ship" && str_or(a.context_backend, "") == "" {
    context_backend = "hf";
}
let model_name = str_or(a.model, "twotower");
// TwoTower forbids free-form compositional tokenizer for production/smoke trains.
let output_tokenizer = str_or(a.output_tokenizer, "choice");
if output_tokenizer != "choice" && output_tokenizer != "lexer" {
    pause("verification", "args.output_tokenizer must be 'choice' or 'lexer' (compositional/free-form is forbidden)");
}
let ship_gates = bool_or(a.ship_gates, false);
if recipe == "ship" { ship_gates = true; }
let matrix_ver = str_or(a.matrix, "v6");
let only_id = str_or(a.only, "E53");
let allow_paid = bool_or(a.allow_paid, false);
let continuous = bool_or(a.continuous, true);
if bool_or(a.once, false) { continuous = false; }
let skip_completed = bool_or(a.skip_completed, true);
let seed = int_or(a.seed, 0);
let force_phases = if a.force_phases == () { [] } else { a.force_phases };
// Include torch extra so sft/eval work without a pre-warmed venv.
// Data-only recipes can override: "uv run python -m scripts.slm"
let cli = str_or(a.cli, "uv run --extra torch python -m scripts.slm");
let source_default = "fixture";
if recipe == "ship" { source_default = "all"; }
if recipe == "matrix" { source_default = "fixture"; }
let source = str_or(a.source, source_default);
let eval_limit_default = 0;
if recipe == "smoke" { eval_limit_default = 3; }
let eval_limit = int_or(a.eval_limit, eval_limit_default);
let publish_data = bool_or(a.publish_data, false);

let planned = [];
if recipe == "custom" {
    planned = filter_known_phases(a.phases);
    if planned.len() == 0 {
        pause("verification", "recipe=custom needs args.phases array of known phase ids");
    }
} else {
    planned = recipe_phases(recipe);
}

if !allow_paid && context_backend == "hf" && !dry_run {
    log("hf backend without allow_paid: force --no-sync-checkpoints; no hf-jobs/remote");
}

let honesty_mode = "fixture_or_scratch";
if ship_gates { honesty_mode = "ship_gates"; }

let state_path = "outputs/autoresearch/" + run_id + "/autotrain-state.json";
let workflow_dir = "outputs/autoresearch/" + run_id;
let run_dir = "outputs/runs/" + run_id;

let cfg = #{
    recipe: recipe,
    run_id: run_id,
    train_version: train_version,
    eval_version: eval_version,
    train_dir: train_dir,
    test_dir: test_dir,
    steps: steps,
    device: device,
    context_backend: context_backend,
    model: model_name,
    output_tokenizer: output_tokenizer,
    ship_gates: ship_gates,
    matrix: matrix_ver,
    only: only_id,
    allow_paid: allow_paid,
    continuous: continuous,
    skip_completed: skip_completed,
    seed: seed,
    honesty_mode: honesty_mode,
    state_path: state_path,
    workflow_dir: workflow_dir,
    run_dir: run_dir,
    dry_run: dry_run,
    cli: cli,
    source: source,
    eval_limit: eval_limit,
    publish_data: publish_data,
};

// Build full deterministic command plan for all planned phases
let command_plan = #{};
for p in planned {
    command_plan[p] = phase_commands(p, cfg);
}

log("autotrain recipe=" + recipe + " run_id=" + run_id + " phases=[" + join_list(planned) + "] cli=" + cli);

let events = [];
let phase_results = [];
let failed = false;
let fail_summary = "";
let contracts = contracts_text();
let wiki = openwiki_block();
let cfg_json = json_encode(cfg);

// ── Plan ─────────────────────────────────────────────────────────────────

phase("Plan");

// Always emit plan to scratch first (observable even if inspect agent fails)
let plan_snap = #{
    config: cfg,
    planned: planned,
    command_plan: command_plan,
    openwiki: [
        "docs/openwiki/quickstart.md",
        "docs/openwiki/workflows/train-eval-docs.md",
        "docs/openwiki/operations/checkpoints-and-agents.md",
    ],
    skill: ".agents/skills/autotrain/SKILL.md",
    cli_map: "python -m scripts.slm list  (≡ uv run python -m scripts.slm list)",
};
let plan_path = write_scratch_file("plan.json", json_encode(plan_snap));
log("wrote plan scratch " + plan_path);

let inspect_prompt = "";
inspect_prompt += "PLAN inspector for autotrain — read-only.\n";
inspect_prompt += wiki;
inspect_prompt += contracts;
inspect_prompt += "\nCONFIG:\n";
inspect_prompt += cfg_json;
inspect_prompt += "\nPLANNED PHASES: ";
inspect_prompt += join_list(planned);
inspect_prompt += "\nCOMMAND PLAN:\n";
inspect_prompt += json_encode(command_plan);
inspect_prompt += "\n\nTASK (use tools):\n";
inspect_prompt += "1. Confirm repo root has scripts/slm.py and docs/openwiki/quickstart.md.\n";
inspect_prompt += "2. Check prereqs: src/apps/openui_bridge/node_modules; root node_modules/@openuidev or agentv runner (root npm ci); torch import under cli.\n";
inspect_prompt += "3. If ";
inspect_prompt += state_path;
inspect_prompt += " exists, read completed phases with status=ok.\n";
inspect_prompt += "4. Check artifact shortcuts for skip: train manifest, test_dir, train_summary.json.\n";
inspect_prompt += "5. Blockers: missing bridge deps; rl without report; allow_paid false but paid path requested.\n";
inspect_prompt += "6. prereqs_ok=true only if scripts/slm.py exists.\n";
inspect_prompt += "Return structured output. Empty completed_phases is valid after checking.\n";

let inspect = agent(inspect_prompt, #{
    label: "plan:inspect",
    capability_mode: "read-only",
    output_schema: inspect_schema,
});

let completed = [];
let blockers = [];
let prereqs_ok = true;
if inspect != () && inspect.success && inspect.output != () {
    if inspect.output.completed_phases != () {
        completed = filter_known_phases(inspect.output.completed_phases);
    }
    if inspect.output.blockers != () && type_of(inspect.output.blockers) == "array" {
        blockers = inspect.output.blockers;
    }
    if inspect.output.prereqs_ok == false {
        prereqs_ok = false;
    }
    log("plan completed=[" + join_list(completed) + "] blockers=" + blockers.len().to_string());
} else {
    log("plan inspect degraded — continuing without skip list");
    events = push_event(events, "plan", "degraded", "inspect failed");
}

if !prereqs_ok && !dry_run {
    await_user("verification", "Plan prereqs_ok=false (missing scripts/slm.py or critical root files). Fix the checkout, then resume.");
}

let skip_set = [];
if skip_completed {
    for p in completed {
        if !contains_str(force_phases, p) {
            skip_set.push(p);
        }
    }
}

let queue = [];
for p in planned {
    if contains_str(skip_set, p) {
        events = push_event(events, p, "skipped", "durable state complete");
        phase_results.push(#{ phase: p, status: "skipped", summary: "skipped — durable complete" });
        log("skip " + p);
    } else {
        queue.push(p);
    }
}

if !dry_run && blockers.len() > 0 {
    let blocker_text = join_list(blockers);
    log("blockers: " + blocker_text);
    if contains_str(queue, "rl") || (allow_paid && context_backend == "hf") {
        await_user("verification", "Plan blockers: " + blocker_text);
    }
}

// Human-readable plan report
let plan_md = "";
plan_md += "# Autotrain command plan\n\n";
plan_md += "- recipe: `" + recipe + "`\n";
plan_md += "- run_id: `" + run_id + "`\n";
plan_md += "- honesty: `" + honesty_mode + "`\n";
plan_md += "- cli: `" + cli + "`\n";
plan_md += "- queue: " + join_list(queue) + "\n";
plan_md += "- skip: " + join_list(skip_set) + "\n";
plan_md += "- train_dir: `" + train_dir + "`\n";
plan_md += "- test_dir: `" + test_dir + "`\n";
plan_md += "- run_dir: `" + run_dir + "`\n";
plan_md += "- state: `" + state_path + "`\n\n";
plan_md += "## Exact commands by phase\n\n";
for p in planned {
    plan_md += "### " + p + "\n```\n";
    plan_md += join_cmds(command_plan[p]);
    plan_md += "\n```\n\n";
}
plan_md += "## OpenWiki\n\n";
plan_md += "- docs/openwiki/quickstart.md\n";
plan_md += "- docs/openwiki/workflows/train-eval-docs.md\n";
plan_md += "- docs/openwiki/operations/checkpoints-and-agents.md\n";
let plan_md_path = write_scratch_file("plan.md", plan_md);

if dry_run {
    phase("Report");
    let report = plan_md;
    report += "\n---\n*dry_run=true — no execute agents after plan inspect.*\n";
    let rpath = write_scratch_file("report.md", report);
    complete(#{
        summary: "dry-run plan only",
        ok: true,
        recipe: recipe,
        run_id: run_id,
        planned: planned,
        queue: queue,
        skipped: skip_set,
        command_plan: command_plan,
        report_path: rpath,
        plan_path: plan_path,
        plan_md_path: plan_md_path,
        honesty_mode: honesty_mode,
        cli: cli,
    });
}

// ── Data parallel ────────────────────────────────────────────────────────

let need_train = contains_str(queue, "data_train");
let need_test = contains_str(queue, "data_test");

if need_train || need_test {
    phase("Data");
    let data_jobs = [];
    let data_ids = [];
    if need_train {
        let cmds = command_plan["data_train"];
        let arts = expected_artifacts("data_train", train_dir, test_dir, run_dir, workflow_dir, state_path);
        data_jobs.push(#{
            prompt: exec_prompt("data_train", cfg_json, cmds, arts, contracts, wiki),
            label: "exec:data_train",
            capability_mode: "execute",
            output_schema: phase_result_schema,
        });
        data_ids.push("data_train");
    }
    if need_test {
        let cmds = command_plan["data_test"];
        let arts = expected_artifacts("data_test", train_dir, test_dir, run_dir, workflow_dir, state_path);
        data_jobs.push(#{
            prompt: exec_prompt("data_test", cfg_json, cmds, arts, contracts, wiki),
            label: "exec:data_test",
            capability_mode: "execute",
            output_schema: phase_result_schema,
        });
        data_ids.push("data_test");
    }
    log("data fan-out n=" + data_jobs.len().to_string());
    let data_results = parallel(data_jobs);
    let di = 0;
    for r in data_results {
        let pid = data_ids[di];
        let st = result_status(r);
        let sm = result_summary(r, pid);
        phase_results.push(#{ phase: pid, status: st, summary: sm });
        events = push_event(events, pid, st, sm);
        log("data " + pid + " → " + st);
        if st == "failed" {
            failed = true;
            fail_summary = sm;
        }
        di += 1;
    }
}

if failed {
    phase("Report");
    let report = "# Autotrain failed at data\n\n" + fail_summary + "\n\n" + plan_md;
    let rpath = write_scratch_file("report.md", report);
    complete(#{
        summary: "failed at data: " + fail_summary,
        ok: false,
        recipe: recipe,
        run_id: run_id,
        events: events,
        phase_results: phase_results,
        report_path: rpath,
        command_plan: command_plan,
        honesty_mode: honesty_mode,
    });
}

// ── sequential phase runner ──────────────────────────────────────────────

fn run_one_phase(phase_id, title, queue, command_plan, cfg_json, train_dir, test_dir, run_dir, workflow_dir, state_path, contracts, wiki, schema) {
    if !contains_str(queue, phase_id) {
        return #{ ran: false };
    }
    phase(title);
    let cmds = command_plan[phase_id];
    let arts = expected_artifacts(phase_id, train_dir, test_dir, run_dir, workflow_dir, state_path);
    let mode = "execute";
    if phase_id == "verify" { mode = "read-only"; }
    let r = agent(
        exec_prompt(phase_id, cfg_json, cmds, arts, contracts, wiki),
        #{
            label: "exec:" + phase_id,
            capability_mode: mode,
            output_schema: schema,
        },
    );
    #{
        ran: true,
        phase: phase_id,
        status: result_status(r),
        summary: result_summary(r, phase_id),
        result: r,
    }
}

// Quality
if contains_str(queue, "quality") {
    let out = run_one_phase("quality", "Quality", queue, command_plan, cfg_json, train_dir, test_dir, run_dir, workflow_dir, state_path, contracts, wiki, phase_result_schema);
    phase_results.push(#{ phase: "quality", status: out.status, summary: out.summary });
    events = push_event(events, "quality", out.status, out.summary);
    log("quality → " + out.status);
    if out.status == "failed" {
        failed = true;
        fail_summary = out.summary;
    }
}

if failed {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# Failed at quality\n\n" + fail_summary + "\n");
    complete(#{ summary: "failed at quality: " + fail_summary, ok: false, recipe: recipe, run_id: run_id, events: events, phase_results: phase_results, report_path: rpath, honesty_mode: honesty_mode });
}

// Train legs
let train_order = ["sft", "distill", "preference", "rl"];
let any_train = false;
for t in train_order {
    if contains_str(queue, t) { any_train = true; }
}
if any_train {
    phase("Train");
    for t in train_order {
        if contains_str(queue, t) {
            if t == "rl" { log("rl fail-closed on readiness report"); }
            let cmds = command_plan[t];
            let arts = expected_artifacts(t, train_dir, test_dir, run_dir, workflow_dir, state_path);
            let r = agent(
                exec_prompt(t, cfg_json, cmds, arts, contracts, wiki),
                #{ label: "exec:" + t, capability_mode: "execute", output_schema: phase_result_schema },
            );
            let st = result_status(r);
            let sm = result_summary(r, t);
            phase_results.push(#{ phase: t, status: st, summary: sm });
            events = push_event(events, t, st, sm);
            log("train " + t + " → " + st);
            if st == "failed" {
                failed = true;
                fail_summary = sm;
                break;
            }
        }
    }
}

if failed {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# Failed during train\n\n" + fail_summary + "\n");
    complete(#{ summary: "failed during train: " + fail_summary, ok: false, recipe: recipe, run_id: run_id, events: events, phase_results: phase_results, report_path: rpath, honesty_mode: honesty_mode });
}

// Eval
if contains_str(queue, "eval") {
    phase("Eval");
    let cmds = command_plan["eval"];
    let arts = expected_artifacts("eval", train_dir, test_dir, run_dir, workflow_dir, state_path);
    let r = agent(
        exec_prompt("eval", cfg_json, cmds, arts, contracts, wiki),
        #{ label: "exec:eval", capability_mode: "execute", output_schema: phase_result_schema },
    );
    let st = result_status(r);
    let sm = result_summary(r, "eval");
    phase_results.push(#{ phase: "eval", status: st, summary: sm });
    events = push_event(events, "eval", st, sm);
    log("eval → " + st);
    if st == "failed" {
        failed = true;
        fail_summary = sm;
    }
}

if failed {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# Failed at eval\n\n" + fail_summary + "\n");
    complete(#{ summary: "failed at eval: " + fail_summary, ok: false, recipe: recipe, run_id: run_id, events: events, phase_results: phase_results, report_path: rpath, honesty_mode: honesty_mode });
}

// Matrix
if contains_str(queue, "matrix") {
    phase("Matrix");
    let cmds = command_plan["matrix"];
    let arts = expected_artifacts("matrix", train_dir, test_dir, run_dir, workflow_dir, state_path);
    let r = agent(
        exec_prompt("matrix", cfg_json, cmds, arts, contracts, wiki),
        #{ label: "exec:matrix", capability_mode: "execute", output_schema: phase_result_schema },
    );
    let st = result_status(r);
    let sm = result_summary(r, "matrix");
    phase_results.push(#{ phase: "matrix", status: st, summary: sm });
    events = push_event(events, "matrix", st, sm);
    log("matrix → " + st);
    if st == "failed" {
        failed = true;
        fail_summary = sm;
    }
}

if failed {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# Failed at matrix\n\n" + fail_summary + "\n");
    complete(#{ summary: "failed at matrix: " + fail_summary, ok: false, recipe: recipe, run_id: run_id, events: events, phase_results: phase_results, report_path: rpath, honesty_mode: honesty_mode });
}

// Closeout
if contains_str(queue, "closeout") {
    phase("Closeout");
    let cmds = command_plan["closeout"];
    let arts = expected_artifacts("closeout", train_dir, test_dir, run_dir, workflow_dir, state_path);
    let r = agent(
        exec_prompt("closeout", cfg_json, cmds, arts, contracts, wiki),
        #{ label: "exec:closeout", capability_mode: "execute", output_schema: phase_result_schema },
    );
    let st = result_status(r);
    let sm = result_summary(r, "closeout");
    phase_results.push(#{ phase: "closeout", status: st, summary: sm });
    events = push_event(events, "closeout", st, sm);
    log("closeout → " + st);
    if st == "failed" {
        failed = true;
        fail_summary = sm;
    }
}

// ── Verify ───────────────────────────────────────────────────────────────

let verified_ok = true;
let verify_confirmed = [];
let verify_rejected = [];

if contains_str(queue, "verify") {
    phase("Verify");
    let harvest = "";
    harvest += "Collect 3-6 verifiable claims for this autotrain run. Read-only tools only.\n";
    harvest += wiki;
    harvest += contracts;
    harvest += "\nCONFIG:\n" + cfg_json + "\n";
    harvest += "PHASE RESULTS:\n" + json_encode(phase_results) + "\n";
    harvest += "COMMAND PLAN:\n" + json_encode(command_plan) + "\n";
    harvest += "Inspect outputs/ and docs/. Claims must cite real evidence_path files.\n";
    harvest += "Include: durable state updated; iron-law docs if train/eval/matrix ran; no ship claim without ship_gates.\n";

    let harvested = agent(harvest, #{
        label: "verify:harvest",
        capability_mode: "read-only",
        output_schema: claim_schema,
    });

    let claims = [];
    if harvested != () && harvested.success && harvested.output != () && harvested.output.claims != () {
        claims = harvested.output.claims;
    }

    if type_of(claims) != "array" || claims.len() == 0 {
        verified_ok = false;
        events = push_event(events, "verify", "failed", "no claims harvested");
        phase_results.push(#{ phase: "verify", status: "failed", summary: "no claims harvested" });
        log("verify fail-closed: no claims");
    } else {
        let use_n = claims.len();
        if use_n > 6 { use_n = 6; }
        let vjobs = [];
        let ci = 0;
        while ci < use_n {
            let c = claims[ci];
            let cid = str_or(c.id, "c" + ci.to_string());
            let vp = "";
            vp += "Adversarially verify claim. Read evidence_path with tools.\n";
            vp += "claim_id: " + cid + "\n";
            vp += "claim: " + str_or(c.claim, "") + "\n";
            vp += "evidence_path: " + str_or(c.evidence_path, "") + "\n";
            vp += "real=true ONLY with independent concrete evidence. Missing file → real=false.\n";
            vp += "Fixture success ≠ ship readiness.\n";
            vjobs.push(#{
                prompt: vp,
                label: "verify:" + cid,
                capability_mode: "read-only",
                output_schema: verdict_schema,
            });
            ci += 1;
        }
        let verdicts = parallel(vjobs);
        let vi = 0;
        for v in verdicts {
            let c = claims[vi];
            let cid = str_or(c.id, "c" + vi.to_string());
            if v != () && v.success && v.output != () && v.output.real == true
                && str_or(v.output.evidence, "") != "" {
                verify_confirmed.push(#{
                    claim_id: cid,
                    claim: str_or(c.claim, ""),
                    evidence: str_or(v.output.evidence, ""),
                });
            } else {
                verified_ok = false;
                let reason = "unverified";
                if v != () && v.success && v.output != () {
                    reason = str_or(v.output.reason, "real=false");
                } else {
                    reason = "verifier failed";
                }
                verify_rejected.push(#{
                    claim_id: cid,
                    claim: str_or(c.claim, ""),
                    reason: reason,
                });
            }
            vi += 1;
        }
        let vst = "failed";
        if verified_ok { vst = "ok"; }
        log("verify " + verify_confirmed.len().to_string() + "/" + use_n.to_string() + " confirmed");
        events = push_event(events, "verify", vst, verify_confirmed.len().to_string() + " confirmed");
        phase_results.push(#{
            phase: "verify",
            status: vst,
            summary: verify_confirmed.len().to_string() + " confirmed / " + verify_rejected.len().to_string() + " rejected",
        });
    }
}

// ── Report + durable persist ─────────────────────────────────────────────

phase("Report");

let ok = !failed && verified_ok;
for pr in phase_results {
    if pr.phase == "closeout" && pr.status == "failed" {
        ok = false;
    }
}

let report = "";
report += "# Autotrain workflow report\n\n";
report += "- ok: **" + ok.to_string() + "**\n";
report += "- recipe: `" + recipe + "`\n";
report += "- run_id: `" + run_id + "`\n";
report += "- honesty_mode: `" + honesty_mode + "`\n";
report += "- cli: `" + cli + "`\n";
report += "- device/steps/backend: `" + device + "` / " + steps.to_string() + " / `" + context_backend + "`\n";
report += "- train_dir: `" + train_dir + "`\n";
report += "- test_dir: `" + test_dir + "`\n";
report += "- run_dir: `" + run_dir + "`\n";
report += "- state: `" + state_path + "`\n\n";
report += "## Phase results\n\n";
for pr in phase_results {
    report += "- `" + str_or(pr.phase, "?") + "` → **" + str_or(pr.status, "?") + "** — " + str_or(pr.summary, "") + "\n";
}
report += "\n## Commands (deterministic plan)\n\n";
for p in planned {
    report += "### " + p + "\n```\n" + join_cmds(command_plan[p]) + "\n```\n";
}
report += "\n## Verification\n\nConfirmed " + verify_confirmed.len().to_string() + " / rejected " + verify_rejected.len().to_string() + "\n";
for c in verify_confirmed {
    report += "- OK `" + str_or(c.claim_id, "") + "`: " + str_or(c.claim, "") + "\n";
}
for c in verify_rejected {
    report += "- NO `" + str_or(c.claim_id, "") + "`: " + str_or(c.claim, "") + " — " + str_or(c.reason, "") + "\n";
}
report += "\n## Honesty\n\nFixture/smoke evidence is wiring only. See docs/openwiki/workflows/train-eval-docs.md and honest-ship-eval.\n";
report += "\n## Continuation\n\n";
if continuous {
    report += "Loop remains active. Commit, fetch+merge origin/main, repair every check, replay Lean certificates, audit typed harness signals, then resume this workflow.\n";
    report += "Between runs: `" + cli + " autoresearch status --loop-id " + run_id + " --matrix --last 5`.\n";
} else {
    report += "Finite `once` pass requested.\n";
}

let rpath = write_scratch_file("report.md", report);
let status_path = write_scratch_file("status.json", json_encode(#{
    ok: ok,
    recipe: recipe,
    run_id: run_id,
    honesty_mode: honesty_mode,
    phase_results: phase_results,
    events: events,
    verify_confirmed: verify_confirmed,
    verify_rejected: verify_rejected,
    command_plan: command_plan,
    config: cfg,
}));

let persist = "";
persist += "Persist durable autotrain state for cross-session resume.\n";
persist += "mkdir -p " + workflow_dir + "\n";
persist += "Write " + state_path + " merging phase statuses from:\n";
persist += json_encode(phase_results);
persist += "\nAlso write " + workflow_dir + "/last_status.json and " + workflow_dir + "/command_plan.json from:\n";
persist += json_encode(#{ ok: ok, recipe: recipe, run_id: run_id, command_plan: command_plan, honesty_mode: honesty_mode });
persist += "\nDo not train. Read-write only for workflow_dir + docs if closeout left gaps.\n";
persist += "Record continuous=" + continuous.to_string() + " and the resume/result-matrix command. Do not mark a continuous loop complete.\n";
persist += "Return one-line summary.\n";

let persist_r = agent(persist, #{
    label: "report:persist",
    capability_mode: "read-write",
});
if persist_r != () && persist_r.success {
    log("durable state persisted");
} else {
    log("warning: persist agent failed; scratch report remains");
}

let b = budget();
log("budget spent=" + b.spent.to_string() + "/" + b.total.to_string());

complete(#{
    summary: if continuous {
        if ok { "autotrain iteration complete; loop remains active" } else { "autotrain iteration failed; loop remains active" }
    } else {
        if ok { "autotrain finite pass completed" } else { "autotrain finite pass completed with failures" }
    },
    active: continuous,
    ok: ok,
    recipe: recipe,
    run_id: run_id,
    honesty_mode: honesty_mode,
    phase_results: phase_results,
    events: events,
    verify_confirmed: verify_confirmed,
    verify_rejected: verify_rejected,
    command_plan: command_plan,
    report_path: rpath,
    status_path: status_path,
    plan_path: plan_path,
    state_path: state_path,
    train_dir: train_dir,
    test_dir: test_dir,
    run_dir: run_dir,
    cli: cli,
    budget: b,
});
