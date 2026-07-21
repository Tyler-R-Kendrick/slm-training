import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { getJSON, usePoll } from "./api";
import { CapsContext, type Caps } from "./caps";
import { Overview } from "./pages/Overview";
import { Data } from "./pages/Data";
import { Experiments } from "./pages/Experiments";
import { Smoke } from "./pages/Smoke";
import { Checkpoints } from "./pages/Checkpoints";
import { Playground } from "./pages/Playground";
import { RunDetail } from "./pages/RunDetail";
import { DslView } from "./interpret/DslView";
import {
  defaultDashboardRenderer,
  initFeatureRuntime,
  trackFeatureExposure,
} from "./features/runtime";

type Nav = (to: string) => void;

// Compiled (hand-written React) twins of the interpreted .openui programs —
// the ◈/◇ toggle switches renderers for the same page.
const COMPILED: Record<string, React.ComponentType<{ navigate: Nav }>> = {
  "/": Overview,
  "/data": Data,
  "/experiments": Experiments,
  "/smoke": Smoke,
  "/checkpoints": Checkpoints,
  "/playground": () => <Playground />,
};

const ROUTES: { path: string; label: string; icon: string }[] = [
  { path: "/", label: "Overview", icon: "◎" },
  { path: "/data", label: "Training Data", icon: "▤" },
  { path: "/experiments", label: "Experiments", icon: "⚗" },
  { path: "/smoke", label: "Smoke Runs", icon: "✷" },
  { path: "/checkpoints", label: "Checkpoints", icon: "◆" },
  { path: "/playground", label: "Playground", icon: "✎" },
];

function useRouter(): [string, Nav] {
  const [path, setPath] = useState(window.location.pathname);
  const pathRef = useRef(path);
  pathRef.current = path;
  const canNavigate = (to: string) => window.dispatchEvent(new CustomEvent("slm-before-navigate", {
    cancelable: true,
    detail: { from: pathRef.current, to },
  }));
  useEffect(() => {
    const onPop = () => {
      const next = window.location.pathname;
      if (!canNavigate(next)) {
        window.history.pushState({}, "", pathRef.current);
        return;
      }
      setPath(next);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const navigate: Nav = (to) => {
    if (to === window.location.pathname) return;
    if (!canNavigate(to)) return;
    window.history.pushState({}, "", to);
    setPath(to);
    window.scrollTo(0, 0);
  };
  return [path, navigate];
}

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => localStorage.getItem("slm-theme") || "dark");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("slm-theme", theme);
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

type Mode = "interpreted" | "compiled";
const MODE_KEY = "slm-mode";

function readStoredMode(): Mode | null {
  const stored = localStorage.getItem(MODE_KEY);
  return stored === "compiled" || stored === "interpreted" ? stored : null;
}

function useMode(featuresReady: boolean): [Mode, (m: Mode) => void] {
  const [mode, setMode] = useState<Mode>(() => readStoredMode() ?? "interpreted");

  useEffect(() => {
    if (!featuresReady) return;
    const stored = readStoredMode();
    if (stored) {
      setMode(stored);
      return;
    }
    setMode(defaultDashboardRenderer());
  }, [featuresReady]);

  useEffect(() => {
    document.documentElement.dataset.mode = mode;
    localStorage.setItem(MODE_KEY, mode);
  }, [mode]);

  const setModeTracked = (next: Mode) => {
    setMode(next);
    if (featuresReady) {
      trackFeatureExposure("dashboard-renderer-selected", { mode: next });
    }
  };
  return [mode, setModeTracked];
}

/** Shell strip: experiment / bucket / jobs freshness (always), cold-start callout when needed. */
function FreshnessBanner() {
  const { data } = usePoll<{
    outputs_present?: boolean;
    freshness?: {
      newest_experiment_date?: string | null;
      experiment_count?: number;
      local_run_count?: number;
      bucket?: { ok?: boolean; count?: number; updated_at?: string | null; error?: string | null };
      hf_jobs?: { ok?: boolean; auth?: boolean; count?: number; error?: string | null };
    };
  }>("/api/system", 30000);
  if (!data) return null;
  const f = data.freshness ?? {};
  const cold = data.outputs_present === false;
  const bucket = f.bucket?.ok
    ? `${f.bucket.count ?? 0} remote${f.bucket.updated_at ? `, updated ${String(f.bucket.updated_at).slice(0, 10)}` : ""}`
    : f.bucket?.error
      ? "unavailable"
      : "—";
  const jobs = f.hf_jobs?.ok
    ? String(f.hf_jobs.count ?? 0)
    : f.hf_jobs?.auth === false
      ? "auth needed"
      : "—";
  return (
    <div className="freshness-banner" role="status">
      <span className={`prov ${cold ? "prov-committed" : "prov-live"}`}>{cold ? "snapshot" : "live"}</span>
      Experiments newest {f.newest_experiment_date ?? "—"}
      {" · "}
      Bucket {bucket}
      {" · "}
      HF Jobs {jobs}
      {cold ? (
        <>
          {" — "}no local <span className="mono">outputs/</span>; remote inventory still listed when available.
        </>
      ) : null}
    </div>
  );
}

function App() {
  const [path, navigate] = useRouter();
  const [theme, toggleTheme] = useTheme();
  const [featuresReady, setFeaturesReady] = useState(false);
  const [mode, setMode] = useMode(featuresReady);
  const [caps, setCaps] = useState<Caps>({ execution: false, read_only: true, jobs: [] });

  useEffect(() => {
    initFeatureRuntime()
      .then(() => setFeaturesReady(true))
      .catch((err) => {
        console.warn("OpenFeature bootstrap failed; using local defaults", err);
        setFeaturesReady(true);
      });
    getJSON<Caps>("/api/capabilities")
      .then(setCaps)
      .catch(() => setCaps({ execution: false, read_only: true, jobs: [] }));
  }, []);

  const runMatch = path.startsWith("/runs/")
    ? decodeURIComponent(path.slice("/runs/".length))
    : null;
  const route = ROUTES.find((r) => r.path === path) ?? ROUTES[0];
  // A run-detail view is reached from Experiments — keep that nav item lit.
  const activePath = runMatch ? "/experiments" : route.path;

  return (
    <CapsContext.Provider value={caps}>
      <div className="app">
        <aside className="sidebar">
          <div className="brand" onClick={() => navigate("/")} style={{ cursor: "pointer" }}>
            <svg className="brand-logo" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="6" fill="var(--bg-3)" />
              <path d="M5 21 L12 12 L17 17 L27 7" stroke="var(--moss-bright)" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div>
              <div className="brand-name">SLM</div>
              <div className="brand-sub">mission control</div>
            </div>
          </div>
          {ROUTES.map((r) => (
            <a
              key={r.path}
              href={r.path}
              aria-current={r.path === activePath ? "page" : undefined}
              className={`nav-link ${r.path === activePath ? "active" : ""}`}
              onClick={(event) => {
                // Only intercept unmodified primary clicks — keep Ctrl/Cmd-click
                // opening these real links in a new tab.
                if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                event.preventDefault();
                navigate(r.path);
              }}
            >
              <span className="nav-icon">{r.icon}</span>
              {r.label}
            </a>
          ))}
          <div className="nav-spacer" />
          {!caps.execution && (
            <div className="hint" style={{ margin: "0 0.4rem 0.5rem" }}>
              Read-only. Serve locally with <span className="mono">serve_playground</span> for the full control plane.
            </div>
          )}
          <div className="mode-toggle" role="group" aria-label="Page renderer">
            {(["compiled", "interpreted"] as Mode[]).map((m) => (
              <button
                key={m}
                className={`mode-btn ${mode === m ? "active" : ""}`}
                aria-pressed={mode === m}
                title={m === "compiled" ? "Hand-written React" : "Live OpenUI Lang program"}
                onClick={() => setMode(m)}
              >
                {m === "compiled" ? "◈ Compiled" : "◇ Interpreted"}
              </button>
            ))}
          </div>
          <div className="hint" style={{ margin: "0 0.4rem 0.5rem" }}>
            Same page, two renderers — the interpreted mode runs the OpenUI DSL this repo trains models to write.
          </div>
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </aside>
        <main className="main">
          <FreshnessBanner />
          {(() => {
            if (runMatch) return <RunDetail runId={runMatch} navigate={navigate} />;
            const Compiled = COMPILED[route.path];
            if (mode === "compiled" && Compiled) return <Compiled key={`${route.path}:compiled`} navigate={navigate} />;
            return <DslView key={`${route.path}:interpreted`} page={route.path} navigate={navigate} />;
          })()}
        </main>
      </div>
    </CapsContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
