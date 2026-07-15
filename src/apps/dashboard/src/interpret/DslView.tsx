// Interpreted-mode page renderer: fetch the page's committed .openui program and
// run it live through the official <Renderer> with our hybrid library, the /api
// toolProvider, and the app action handler — in the app's current theme.
import React, { useEffect, useState } from "react";
import { Renderer } from "@openuidev/react-lang";
import { ThemeProvider } from "@openuidev/react-ui";
import "@openuidev/react-ui/index.css";
import { dashboardLibrary } from "./library";
import { toolProvider } from "./toolProvider";
import { makeOnAction } from "./actions";
import { navRef } from "./nav";
import { ErrorNote } from "../components";

export function DslView({ page, navigate }: { page: string; navigate: (to: string) => void }) {
  const [src, setSrc] = useState<string | null>(null);
  const [fetchErr, setFetchErr] = useState<string | null>(null);
  const [parseErrs, setParseErrs] = useState<string[]>([]);
  const theme = (document.documentElement.dataset.theme as "light" | "dark") || "dark";
  const slug = page === "/" ? "overview" : page.replace(/^\//, "");
  // Let custom library components (NavChip, JobList) drive in-app navigation.
  navRef.current = navigate;

  useEffect(() => {
    setSrc(null);
    setFetchErr(null);
    setParseErrs([]);
    fetch(`/static/openui/${slug}.openui`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status} ${slug}.openui`))))
      .then(setSrc)
      .catch((e) => setFetchErr(String(e?.message ?? e)));
  }, [slug]);

  if (fetchErr) return <ErrorNote error={`No DSL for this page: ${fetchErr}`} />;
  if (src == null) return <div className="loading">Loading DSL…</div>;

  const onAction = makeOnAction(navigate);

  return (
    <div className="dsl-view">
      {parseErrs.length > 0 && (
        <div className="error-note">
          DSL parse issues: {parseErrs.slice(0, 4).join(" · ")}
        </div>
      )}
      <ThemeProvider mode={theme}>
        <Renderer
          response={src}
          library={dashboardLibrary}
          isStreaming={false}
          toolProvider={toolProvider}
          onAction={onAction}
          onError={(errs: any[]) => {
            setParseErrs((errs || []).map((e) => String(e?.message ?? e)).filter(Boolean));
          }}
        />
      </ThemeProvider>
    </div>
  );
}
