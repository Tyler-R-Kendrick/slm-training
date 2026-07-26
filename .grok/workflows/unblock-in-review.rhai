// unblock-in-review — drain Linear "In Review" with max parallelism + self-heal
//
// Discovers In Review issues (oldest → newest), claims them for Grok, reuses
// existing open PRs when present, fans out isolated worktree workers, then
// runs a self-heal pass on failures before reporting.
//
// Args (object; all optional):
//   team              Linear team (default "slm-training")
//   state             Linear state name (default "In Review")
//   max_issues        max issues to claim+work this run (default 3, max 8)
//   dry_run           discover+plan only; no claims/worktrees/PRs
//   base_branch       git base (default "main")
//   claim_label       label applied on claim (default "claimed-by-grok")
//   done_state        Linear state after successful merge (default "Done")
//   include_claimed   re-pick already claim_label issues (default false)
//   issue_ids         optional explicit ids (e.g. ["SLM-300"])
//   repo              "owner/name" (default "Tyler-R-Kendrick/slm-training")
//   worktree_parent   default: absolute sibling ../.worktrees-unblock
//   allow_merge       squash-merge when checks green (default true)
//   max_check_rounds  CI fix loops per worker (default 3)
//   heal              run self-heal pass on non-ok workers (default true)
//   prefer_existing_pr  babysit open PRs for the issue when found (default true)
//
// Robustness / self-heal:
//   - Discover retries once on agent failure
//   - Prefer existing open PR (gh) over greenfield branch when present
//   - Claim is idempotent: reuse worktree/branch; create label if missing
//   - Workers recover dirty worktrees, rebase conflicts, CI failures
//   - Heal pass re-runs failed/blocked workers once with prior error context
//   - Durable notes under outputs/unblock-in-review/ (best-effort)
//
// Budget: ~3 + 2*max_issues  (discover[+retry], claim, workers, heal, report)
// Recommended agent_budget: 48 (max_issues=3) or 96 (max_issues=8)
// Max single parallel panel: max_issues

let meta = #{
    name: "unblock-in-review",
    description: "Claim oldest Linear In Review issues, finish each in an isolated worktree (reuse open PRs), fix CI, squash-merge, self-heal failures, clean up — in parallel",
    when_to_use: "When the Linear In Review queue is backed up and you want Grok to claim, complete, PR-babysit, merge, and self-heal multiple issues concurrently",
    phases: [
        #{ title: "Discover", detail: "list In Review + open PRs, oldest first" },
        #{ title: "Claim", detail: "label/comment claim; idempotent worktrees" },
        #{ title: "Work", detail: "parallel finish/PR/merge workers" },
        #{ title: "Heal", detail: "retry failed/blocked workers once" },
        #{ title: "Report", detail: "scoreboard + leftovers" },
    ],
};

// ── schemas ──────────────────────────────────────────────────────────────

let discover_schema = #{
    "type": "object",
    "required": ["issues", "total_in_review", "skipped"],
    "properties": #{
        "total_in_review": #{ "type": "integer" },
        "issues": #{
            "type": "array", "maxItems": 16,
            "items": #{
                "type": "object",
                "required": ["id", "title", "url", "created_at", "branch_hint"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "url": #{ "type": "string" },
                    "created_at": #{ "type": "string" },
                    "branch_hint": #{ "type": "string" },
                    "priority": #{ "type": "string" },
                    "assignee": #{ "type": "string" },
                    "project": #{ "type": "string" },
                    "already_claimed": #{ "type": "boolean" },
                    "existing_pr_number": #{ "type": "integer" },
                    "existing_pr_url": #{ "type": "string" },
                    "existing_pr_head": #{ "type": "string" },
                    "mode": #{ "type": "string" },
                    "summary": #{ "type": "string" },
                },
            },
        },
        "skipped": #{
            "type": "array", "maxItems": 24,
            "items": #{
                "type": "object",
                "required": ["id", "reason"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "reason": #{ "type": "string" },
                },
            },
        },
        "notes": #{ "type": "string" },
    },
};

let claim_schema = #{
    "type": "object",
    "required": ["claimed"],
    "properties": #{
        "claimed": #{
            "type": "array", "maxItems": 16,
            "items": #{
                "type": "object",
                "required": ["id", "branch", "worktree", "status"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "url": #{ "type": "string" },
                    "branch": #{ "type": "string" },
                    "worktree": #{ "type": "string" },
                    "status": #{ "type": "string" },
                    "mode": #{ "type": "string" },
                    "existing_pr_number": #{ "type": "integer" },
                    "existing_pr_url": #{ "type": "string" },
                    "reused_worktree": #{ "type": "boolean" },
                    "error": #{ "type": "string" },
                    "summary": #{ "type": "string" },
                },
            },
        },
        "label_ensured": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let worker_schema = #{
    "type": "object",
    "required": ["id", "status", "summary"],
    "properties": #{
        "id": #{ "type": "string" },
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "worktree": #{ "type": "string" },
        "pr_url": #{ "type": "string" },
        "pr_number": #{ "type": "integer" },
        "merge_sha": #{ "type": "string" },
        "linear_done": #{ "type": "boolean" },
        "worktree_cleaned": #{ "type": "boolean" },
        "checks_fixed": #{ "type": "integer" },
        "healed": #{ "type": "boolean" },
        "mode": #{ "type": "string" },
        "error": #{ "type": "string" },
        "next_action": #{ "type": "string" },
        "commits": #{
            "type": "array", "maxItems": 8,
            "items": #{ "type": "string" },
        },
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

fn join_ids(arr) {
    if type_of(arr) != "array" || arr.len() == 0 { return "(none)"; }
    let s = "";
    let i = 0;
    for x in arr {
        if i > 0 { s += ", "; }
        if type_of(x) == "map" {
            s += str_or(x.id, "?");
        } else {
            s += x.to_string();
        }
        i += 1;
    }
    s
}

fn clamp_max(n, lo, hi) {
    if n < lo { return lo; }
    if n > hi { return hi; }
    n
}

fn status_of(o) {
    str_or(o.status, "failed")
}

fn is_ok_status(st) {
    st == "ok" || st == "merged" || st == "pr_ready"
}

fn needs_heal(st) {
    st == "failed" || st == "blocked" || st == "needs_heal"
}

fn worker_prompt(w, cfg_json, heal_hint) {
    let wid = str_or(w.id, "unknown");
    let wbranch = str_or(w.branch, "");
    let wtree = str_or(w.worktree, "");
    let wurl = str_or(w.url, "");
    let wtitle = str_or(w.title, "");
    let mode = str_or(w.mode, "implement");
    let epr = str_or(w.existing_pr_url, "");
    let eprn = int_or(w.existing_pr_number, 0);

    let p = "";
    p += "You are a FULL-LIFECYCLE self-healing worker for ONE Linear issue.\n";
    p += "Issue: ";
    p += wid;
    p += " — ";
    p += wtitle;
    p += "\nURL: ";
    p += wurl;
    p += "\nBranch: ";
    p += wbranch;
    p += "\nWorktree: ";
    p += wtree;
    p += "\nMode: ";
    p += mode;
    p += " (babysit=existing PR; implement=greenfield)\n";
    if epr != "" {
        p += "Existing PR: ";
        p += epr;
        p += " (#";
        p += eprn.to_string();
        p += ")\n";
    }
    p += "\nCONFIG JSON:\n";
    p += cfg_json;
    p += "\n";
    if heal_hint != "" {
        p += "\nSELF-HEAL CONTEXT (prior failure — fix this):\n";
        p += heal_hint;
        p += "\n";
    }
    p += "\nSELF-HEALING RULES (battle-tested):\n";
    p += "- Idempotent: inspect git/Linear/PR state before repeating side effects.\n";
    p += "- STACKED BASE FIX: if PR baseRefName != config.base_branch, retarget with\n";
    p += "  `gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -f base=<base_branch>`.\n";
    p += "  If PR is CLOSED unmerged, open a NEW PR to base_branch from the same head.\n";
    p += "- Worktree create: ALWAYS `git worktree add -B <head> <path> origin/<head>`\n";
    p += "  (never bare `git worktree add path origin/head` — that detaches HEAD).\n";
    p += "- If worktree dirty: commit intentional CI fixes or stash; never leave half-rebase.\n";
    p += "- Rebase onto origin/<base_branch>; resolve conflicts carefully; push --force-with-lease.\n";
    p += "- Required CI: python, python-static, data-build. Vercel/CodeRabbit alone MUST NOT block merge.\n";
    p += "- On CI fail: read `gh run view --log-failed`, fix root cause, push, re-watch. Cap max_check_rounds.\n";
    p += "- Merge: `gh pr merge <n> --squash --delete-branch` then `--admin` if only non-required checks fail.\n";
    p += "- After merge: Linear save_issue state=done_state + comment with PR URL. status=ok only if merged\n";
    p += "  when allow_merge=true (or pr_ready when allow_merge=false and checks green).\n";
    p += "- Clean: git worktree remove --force; never leave orphan worktrees.\n";
    p += "- Never weaken ship gates / delete tests to go green.\n";
    p += "\nSTEPS (execute all):\n";
    p += "0. cd worktree (or create it). git fetch origin --prune.\n";
    p += "1. Linear get issue; gh pr view (number from existing_pr or search title for id).\n";
    p += "2. If PR base != base_branch → PATCH base to main. If closed unmerged → gh pr create --base main --head <head>.\n";
    p += "3. Rebase origin/main in worktree; push force-with-lease; fix CI until required green.\n";
    p += "4. If allow_merge: squash merge. Confirm MERGED via gh pr view.\n";
    p += "5. Linear Done + comment. Remove claim_label if present.\n";
    p += "6. Remove worktree. status ok|pr_ready|blocked|failed|needs_heal with error/next_action.\n";
    p += "Return structured worker_schema.\n";
    p
}

// ── args ─────────────────────────────────────────────────────────────────

let a = args;
if a == () { a = #{}; }

let team = str_or(a.team, "slm-training");
let state_name = str_or(a.state, "In Review");
let max_issues = clamp_max(int_or(a.max_issues, 3), 1, 8);
let dry_run = bool_or(a.dry_run, false);
let base_branch = str_or(a.base_branch, "main");
let claim_label = str_or(a.claim_label, "claimed-by-grok");
let done_state = str_or(a.done_state, "Done");
let include_claimed = bool_or(a.include_claimed, false);
let repo = str_or(a.repo, "Tyler-R-Kendrick/slm-training");
let worktree_parent = str_or(a.worktree_parent, "../.worktrees-unblock");
let allow_merge = bool_or(a.allow_merge, true);
let max_check_rounds = clamp_max(int_or(a.max_check_rounds, 3), 1, 6);
let heal = bool_or(a.heal, true);
let prefer_existing_pr = bool_or(a.prefer_existing_pr, true);
let issue_ids = if a.issue_ids == () { [] } else { a.issue_ids };

let cfg = #{
    team: team,
    state: state_name,
    max_issues: max_issues,
    dry_run: dry_run,
    base_branch: base_branch,
    claim_label: claim_label,
    done_state: done_state,
    include_claimed: include_claimed,
    repo: repo,
    worktree_parent: worktree_parent,
    allow_merge: allow_merge,
    max_check_rounds: max_check_rounds,
    heal: heal,
    prefer_existing_pr: prefer_existing_pr,
    issue_ids: issue_ids,
};

log("unblock-in-review team=" + team + " max=" + max_issues.to_string() + " dry_run=" + dry_run.to_string() + " heal=" + heal.to_string());

let cfg_json = json_encode(cfg);
write_scratch_file("plan.json", cfg_json);

// ── Discover (with one retry) ────────────────────────────────────────────

phase("Discover");

let discover_prompt = "";
discover_prompt += "You DISCOVER Linear In Review issues and match open GitHub PRs.\n";
discover_prompt += "Tools: Linear MCP (list_issues), gh CLI (pr list), read_file as needed.\n";
discover_prompt += "Do NOT claim, edit code, merge, or comment on Linear.\n\n";
discover_prompt += "CONFIG:\n";
discover_prompt += cfg_json;
discover_prompt += "\n\nTASK:\n";
discover_prompt += "1. list_issues state=config.state team=config.team orderBy=createdAt includeArchived=false limit=50.\n";
discover_prompt += "2. Sort OLDEST→NEWEST by createdAt (reverse if API returns newest first).\n";
discover_prompt += "3. If issue_ids non-empty, keep only those ids.\n";
discover_prompt += "4. Skip claim_label unless include_claimed. Skip archived/canceled.\n";
discover_prompt += "5. gh pr list --state open --limit 50 --json number,title,headRefName,url for config.repo.\n";
discover_prompt += "   Match PRs whose title or head contains the issue id (e.g. SLM-300). If multiple, pick highest PR number.\n";
discover_prompt += "6. For each kept issue set mode:\n";
discover_prompt += "   - babysit if prefer_existing_pr and a matching open PR exists\n";
discover_prompt += "   - implement otherwise\n";
discover_prompt += "7. Cap to max_issues after sort/filter. Return issues + skipped + total_in_review.\n";
discover_prompt += "branch_hint: prefer existing_pr_head when present; else Linear gitBranchName; else id slug.\n";
discover_prompt += "existing_pr_head MUST be the GitHub headRefName (often differs from Linear gitBranchName).\n";
discover_prompt += "Empty issues only after a real empty list_issues result.\n";

let discovered = agent(discover_prompt, #{
    label: "discover:linear",
    capability_mode: "execute",
    output_schema: discover_schema,
});

if discovered == () || !discovered.success || discovered.output == () {
    log("discover failed — retrying once");
    discovered = agent(discover_prompt + "\nRETRY: previous attempt failed; re-run list_issues and gh pr list carefully.\n", #{
        label: "discover:retry",
        capability_mode: "execute",
        output_schema: discover_schema,
    });
}

let candidates = [];
let total_in_review = 0;
let skipped = [];
if discovered != () && discovered.success && discovered.output != () {
    if discovered.output.issues != () && type_of(discovered.output.issues) == "array" {
        candidates = discovered.output.issues;
    }
    total_in_review = int_or(discovered.output.total_in_review, candidates.len());
    if discovered.output.skipped != () && type_of(discovered.output.skipped) == "array" {
        skipped = discovered.output.skipped;
    }
    log("discover: " + candidates.len().to_string() + " candidates (≈" + total_in_review.to_string() + " in review)");
} else {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# Discover failed after retry\n");
    complete(#{ summary: "discover failed", ok: false, report_path: rpath, config: cfg });
}

// Script-side hard cap
if candidates.len() > max_issues {
    log("cap candidates " + candidates.len().to_string() + "→" + max_issues.to_string());
    let trimmed = [];
    let ti = 0;
    while ti < max_issues {
        trimmed.push(candidates[ti]);
        ti += 1;
    }
    candidates = trimmed;
}

// Normalize mode field into a fresh array (Rhai for-in maps are by-value)
let normalized = [];
for c in candidates {
    let m = str_or(c.mode, "");
    if m == "" {
        if prefer_existing_pr && int_or(c.existing_pr_number, 0) > 0 {
            m = "babysit";
        } else {
            m = "implement";
        }
    }
    normalized.push(#{
        id: str_or(c.id, ""),
        title: str_or(c.title, ""),
        url: str_or(c.url, ""),
        created_at: str_or(c.created_at, ""),
        branch_hint: str_or(c.branch_hint, ""),
        priority: str_or(c.priority, ""),
        assignee: str_or(c.assignee, ""),
        project: str_or(c.project, ""),
        already_claimed: bool_or(c.already_claimed, false),
        existing_pr_number: int_or(c.existing_pr_number, 0),
        existing_pr_url: str_or(c.existing_pr_url, ""),
        existing_pr_head: str_or(c.existing_pr_head, ""),
        mode: m,
        summary: str_or(c.summary, ""),
    });
}
candidates = normalized;

let disc_path = write_scratch_file("discovered.json", json_encode(#{
    candidates: candidates,
    skipped: skipped,
    total_in_review: total_in_review,
}));

if candidates.len() == 0 {
    phase("Report");
    let report = "# Unblock In Review\n\nNo eligible issues.\n";
    let rpath = write_scratch_file("report.md", report);
    complete(#{
        summary: "no eligible issues",
        ok: true,
        candidates: [],
        report_path: rpath,
        discovered_path: disc_path,
        config: cfg,
    });
}

if dry_run {
    phase("Report");
    let report = "# Unblock In Review — dry run\n\n";
    report += "Queue (oldest→newest), max=";
    report += max_issues.to_string();
    report += ":\n\n";
    for c in candidates {
        report += "- `";
        report += str_or(c.id, "?");
        report += "` **";
        report += str_or(c.mode, "?");
        report += "** — ";
        report += str_or(c.title, "");
        if str_or(c.existing_pr_url, "") != "" {
            report += "\n  - existing PR: ";
            report += str_or(c.existing_pr_url, "");
        }
        report += "\n";
    }
    report += "\nNo claims/worktrees/PRs (`dry_run=true`).\n";
    let rpath = write_scratch_file("report.md", report);
    complete(#{
        summary: "dry-run: " + candidates.len().to_string() + " queued",
        ok: true,
        dry_run: true,
        candidates: candidates,
        report_path: rpath,
        discovered_path: disc_path,
        config: cfg,
    });
}

// ── Claim (idempotent) ───────────────────────────────────────────────────

phase("Claim");

let claim_prompt = "";
claim_prompt += "You CLAIM Linear issues and prepare idempotent git worktrees.\n";
claim_prompt += "Tools: Linear MCP + git. CONFIG:\n";
claim_prompt += cfg_json;
claim_prompt += "\nCANDIDATES:\n";
claim_prompt += json_encode(candidates);
claim_prompt += "\n\nIDEMPOTENT CLAIM STEPS:\n";
claim_prompt += "1. Ensure label claim_label exists (create team label if missing).\n";
claim_prompt += "2. mkdir -p worktree_parent.\n";
claim_prompt += "3. git fetch origin --prune.\n";
claim_prompt += "4. For each candidate:\n";
claim_prompt += "   a. Read issue; merge claim_label into labels (read existing labels first; save_issue labels replaces full set).\n";
claim_prompt += "   b. Comment once if not already commented this hour: claimed by Grok unblock-in-review.\n";
claim_prompt += "   c. Branch naming:\n";
claim_prompt += "      - babysit with existing_pr_head: use that head branch (do not rename remote).\n";
claim_prompt += "      - else: grok/unblock-<id-lower> e.g. grok/unblock-slm-300\n";
claim_prompt += "   d. Worktree path: {worktree_parent}/<id-lower>\n";
claim_prompt += "   e. If worktree exists and is a valid git worktree: REUSE (reused_worktree=true). git -C path fetch; checkout branch.\n";
claim_prompt += "   f. Else if path exists but broken: git worktree remove --force or rm -rf then recreate.\n";
claim_prompt += "   g. Create worktrees (CRITICAL — avoid detached HEAD):\n";
claim_prompt += "      - greenfield: `git worktree add -b grok/unblock-<id-lower> <path> origin/<base>`\n";
claim_prompt += "      - babysit PR head: `git worktree add -B <head> <path> origin/<head>`\n";
claim_prompt += "        NEVER `git worktree add <path> origin/<head>` alone (that detaches HEAD).\n";
claim_prompt += "      Prefer existing_pr_head over Linear gitBranchName when they differ.\n";
claim_prompt += "   h. status=ok|skipped|failed. Never leave partial claim without recording error.\n";
claim_prompt += "5. Return claimed[] for every candidate.\n";
claim_prompt += "Do NOT implement issue bodies yet.\n";

let claimed_r = agent(claim_prompt, #{
    label: "claim:prep",
    capability_mode: "all",
    output_schema: claim_schema,
});

// Claim retry once
if claimed_r == () || !claimed_r.success || claimed_r.output == () {
    log("claim failed — retry once");
    claimed_r = agent(claim_prompt + "\nRETRY: be more careful with worktree reuse; list git worktree list first.\n", #{
        label: "claim:retry",
        capability_mode: "all",
        output_schema: claim_schema,
    });
}

let claimed = [];
if claimed_r != () && claimed_r.success && claimed_r.output != () && claimed_r.output.claimed != () {
    claimed = claimed_r.output.claimed;
} else {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# Claim failed after retry\n");
    complete(#{ summary: "claim failed", ok: false, candidates: candidates, report_path: rpath, config: cfg });
}

write_scratch_file("claimed.json", json_encode(claimed));

// Enrich claimed with mode/pr from candidates when missing
let workers = [];
for c in claimed {
    if status_of(c) == "ok" {
        // copy mode from candidate match
        for cand in candidates {
            if str_or(cand.id, "") == str_or(c.id, "") {
                if str_or(c.mode, "") == "" { c.mode = str_or(cand.mode, "implement"); }
                if int_or(c.existing_pr_number, 0) == 0 {
                    c.existing_pr_number = int_or(cand.existing_pr_number, 0);
                    c.existing_pr_url = str_or(cand.existing_pr_url, "");
                }
                if str_or(c.title, "") == "" { c.title = str_or(cand.title, ""); }
                if str_or(c.url, "") == "" { c.url = str_or(cand.url, ""); }
            }
        }
        workers.push(c);
    } else {
        log("skip worker " + str_or(c.id, "?") + " claim=" + status_of(c));
    }
}

if workers.len() == 0 {
    phase("Report");
    let rpath = write_scratch_file("report.md", "# No successful claims\n");
    complete(#{ summary: "nothing claimed", ok: false, claimed: claimed, report_path: rpath, config: cfg });
}

log("work fan-out n=" + workers.len().to_string() + " → " + join_ids(workers));

// ── Work parallel ────────────────────────────────────────────────────────

phase("Work");

let jobs = [];
for w in workers {
    jobs.push(#{
        prompt: worker_prompt(w, cfg_json, ""),
        label: "work:" + str_or(w.id, "?"),
        capability_mode: "all",
        output_schema: worker_schema,
    });
}
let results = parallel(jobs);

let outcomes = [];
let ri = 0;
for r in results {
    let wid = str_or(workers[ri].id, "unknown");
    if r != () && r.success && r.output != () {
        let o = r.output;
        if str_or(o.id, "") == "" { o.id = wid; }
        outcomes.push(o);
        log("worker " + wid + " → " + status_of(o));
    } else {
        outcomes.push(#{
            id: wid,
            status: "failed",
            summary: "worker agent failed",
            error: "empty or success=false",
            next_action: "heal pass",
            branch: str_or(workers[ri].branch, ""),
            worktree: str_or(workers[ri].worktree, ""),
        });
        log("worker " + wid + " → failed(agent)");
    }
    ri += 1;
}

// ── Heal pass ────────────────────────────────────────────────────────────

if heal {
    let heal_targets = [];
    let hi = 0;
    for o in outcomes {
        if needs_heal(status_of(o)) {
            // find matching worker claim for branch/worktree
            let wmeta = workers[0];
            let found = false;
            for w in workers {
                if str_or(w.id, "") == str_or(o.id, "") {
                    wmeta = w;
                    found = true;
                }
            }
            if found {
                // attach prior error into a copy-like map for prompt
                wmeta.error = str_or(o.error, status_of(o));
                wmeta.prior_summary = str_or(o.summary, "");
                wmeta.prior_status = status_of(o);
                heal_targets.push(#{
                    worker: wmeta,
                    prior: o,
                });
            }
        }
        hi += 1;
    }

    if heal_targets.len() > 0 {
        phase("Heal");
        log("heal fan-out n=" + heal_targets.len().to_string());
        let hjobs = [];
        for t in heal_targets {
            let hint = "";
            hint += "prior_status=";
            hint += str_or(t.prior.status, "");
            hint += " error=";
            hint += str_or(t.prior.error, "");
            hint += " summary=";
            hint += str_or(t.prior.summary, "");
            hint += " next_action=";
            hint += str_or(t.prior.next_action, "");
            hjobs.push(#{
                prompt: worker_prompt(t.worker, cfg_json, hint),
                label: "heal:" + str_or(t.worker.id, "?"),
                capability_mode: "all",
                output_schema: worker_schema,
            });
        }
        let hresults = parallel(hjobs);
        // Replace outcomes for healed ids
        let hj = 0;
        for hr in hresults {
            let tid = str_or(heal_targets[hj].worker.id, "");
            if hr != () && hr.success && hr.output != () {
                let ho = hr.output;
                if str_or(ho.id, "") == "" { ho.id = tid; }
                ho.healed = true;
                // rebuild outcomes array replacing matching id
                let new_out = [];
                for o in outcomes {
                    if str_or(o.id, "") == tid {
                        new_out.push(ho);
                    } else {
                        new_out.push(o);
                    }
                }
                outcomes = new_out;
                log("heal " + tid + " → " + status_of(ho));
            } else {
                log("heal " + tid + " → still failed");
            }
            hj += 1;
        }
    } else {
        log("heal: nothing to heal");
    }
}

write_scratch_file("outcomes.json", json_encode(outcomes));

// ── Report ───────────────────────────────────────────────────────────────

phase("Report");

let ok_n = 0;
let fail_n = 0;
let blocked_n = 0;
let ready_n = 0;
for o in outcomes {
    let st = status_of(o);
    if is_ok_status(st) {
        if st == "pr_ready" { ready_n += 1; } else { ok_n += 1; }
    } else if st == "blocked" {
        blocked_n += 1;
    } else {
        fail_n += 1;
    }
}

let report = "";
report += "# Unblock In Review — report\n\n";
report += "- team/state: `";
report += team;
report += "` / `";
report += state_name;
report += "`\n";
report += "- candidates: ";
report += candidates.len().to_string();
report += " · workers: ";
report += workers.len().to_string();
report += " · total_in_review≈";
report += total_in_review.to_string();
report += "\n";
report += "- merged/ok: **";
report += ok_n.to_string();
report += "** · pr_ready: **";
report += ready_n.to_string();
report += "** · blocked: **";
report += blocked_n.to_string();
report += "** · failed: **";
report += fail_n.to_string();
report += "**\n";
report += "- allow_merge: ";
report += allow_merge.to_string();
report += " · heal: ";
report += heal.to_string();
report += "\n\n## Outcomes\n\n";
for o in outcomes {
    report += "### ";
    report += str_or(o.id, "?");
    report += " — **";
    report += status_of(o);
    report += "**";
    if o.healed == true { report += " (healed)"; }
    report += "\n";
    report += str_or(o.summary, "");
    report += "\n";
    if str_or(o.pr_url, "") != "" {
        report += "- PR: ";
        report += str_or(o.pr_url, "");
        report += "\n";
    }
    if str_or(o.merge_sha, "") != "" {
        report += "- merge: `";
        report += str_or(o.merge_sha, "");
        report += "`\n";
    }
    if str_or(o.error, "") != "" {
        report += "- error: ";
        report += str_or(o.error, "");
        report += "\n";
    }
    if str_or(o.next_action, "") != "" {
        report += "- next: ";
        report += str_or(o.next_action, "");
        report += "\n";
    }
    report += "\n";
}
report += "## Re-run\n\n";
report += "```\n/workflow unblock-in-review {\"max_issues\":3}\n/workflow unblock-in-review {\"dry_run\":true}\n```\n";

let rpath = write_scratch_file("report.md", report);

// Best-effort durable mirror via small agent
let persist = agent(
    "Write outputs/unblock-in-review/latest/outcomes.json and report.md copies from this JSON if possible (mkdir -p). Data:\n" + json_encode(#{ outcomes: outcomes, config: cfg }) + "\nReturn ok.\n",
    #{ label: "report:persist", capability_mode: "read-write" },
);
if persist != () && persist.success {
    log("persisted durable outputs/unblock-in-review/latest");
}

let b = budget();
log("budget " + b.spent.to_string() + "/" + b.total.to_string());

let all_ok = fail_n == 0 && blocked_n == 0;
complete(#{
    summary: ok_n.to_string() + " ok / " + ready_n.to_string() + " pr_ready / " + blocked_n.to_string() + " blocked / " + fail_n.to_string() + " failed",
    ok: all_ok,
    ok_count: ok_n,
    pr_ready_count: ready_n,
    blocked_count: blocked_n,
    failed_count: fail_n,
    outcomes: outcomes,
    claimed: claimed,
    candidates: candidates,
    report_path: rpath,
    discovered_path: disc_path,
    config: cfg,
    budget: b,
});
