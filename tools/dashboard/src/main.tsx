import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { getJSON } from "./api";
import { CapsContext, type Caps } from "./caps";
import { Overview } from "./pages/Overview";
import { Data } from "./pages/Data";
import { Experiments } from "./pages/Experiments";
import { Smoke } from "./pages/Smoke";
import { Checkpoints } from "./pages/Checkpoints";

type Nav = (to: string) => void;

const ROUTES: { path: string; label: string; icon: string; el: (nav: Nav) => React.ReactNode }[] = [
  { path: "/", label: "Overview", icon: "◎", el: (nav) => <Overview navigate={nav} /> },
  { path: "/data", label: "Training Data", icon: "▤", el: (nav) => <Data navigate={nav} /> },
  { path: "/experiments", label: "Experiments", icon: "⚗", el: (nav) => <Experiments navigate={nav} /> },
  { path: "/smoke", label: "Smoke Runs", icon: "✷", el: (nav) => <Smoke navigate={nav} /> },
  { path: "/checkpoints", label: "Checkpoints", icon: "◆", el: (nav) => <Checkpoints navigate={nav} /> },
];

function useRouter(): [string, Nav] {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const navigate: Nav = (to) => {
    if (to === window.location.pathname) return;
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

function App() {
  const [path, navigate] = useRouter();
  const [theme, toggleTheme] = useTheme();
  const [caps, setCaps] = useState<Caps>({ execution: false, read_only: true, jobs: [] });

  useEffect(() => {
    getJSON<Caps>("/api/capabilities")
      .then(setCaps)
      .catch(() => setCaps({ execution: false, read_only: true, jobs: [] }));
  }, []);

  const route = ROUTES.find((r) => r.path === path) ?? ROUTES[0];

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
              className={`nav-link ${r.path === route.path ? "active" : ""}`}
              onClick={() => navigate(r.path)}
            >
              <span className="nav-icon">{r.icon}</span>
              {r.label}
            </a>
          ))}
          <a className="nav-link" href="/playground">
            <span className="nav-icon">✎</span>
            Playground
          </a>
          <div className="nav-spacer" />
          {!caps.execution && (
            <div className="hint" style={{ margin: "0 0.4rem 0.5rem" }}>
              Read-only. Serve locally with <span className="mono">serve_playground</span> for the full control plane.
            </div>
          )}
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </aside>
        <main className="main">{route.el(navigate)}</main>
      </div>
    </CapsContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
