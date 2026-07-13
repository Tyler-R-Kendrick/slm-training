import React, { StrictMode, useMemo, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Renderer } from "@openuidev/react-lang";
import { ThemeProvider, openuiLibrary } from "@openuidev/react-ui";
import "@openuidev/react-ui/defaults.css";
import "@openuidev/react-ui/index.css";

type MountOptions = {
  /** Raw OpenUI Lang source */
  source: string | null | undefined;
  /** When true, keep placeholders as-is; default substitutes demo copy */
  keepPlaceholders?: boolean;
};

const PLACEHOLDER_RE = /:([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)/g;

function humanizePlaceholder(token: string): string {
  const parts = token.replace(/^:/, "").split(".");
  const last = parts[parts.length - 1] || "value";
  const words = last
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim();
  if (!words) return "Sample";
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Substitute :placeholders with readable demo labels so the UI looks like the OpenUI demo. */
export function fillPlaceholders(source: string): string {
  return source.replace(PLACEHOLDER_RE, (match) => humanizePlaceholder(match));
}

function PreviewApp({ source, keepPlaceholders }: MountOptions) {
  const [parseOk, setParseOk] = useState(true);
  const [errors, setErrors] = useState<string[]>([]);
  const response = useMemo(() => {
    const raw = (source || "").trim();
    if (!raw) return null;
    return keepPlaceholders ? raw : fillPlaceholders(raw);
  }, [source, keepPlaceholders]);

  if (!response) {
    return <div className="openui-preview-empty">No OpenUI yet</div>;
  }

  return (
    <ThemeProvider mode="light">
      <div className="openui-preview-root" data-parse-ok={parseOk ? "1" : "0"}>
        {!parseOk && (
          <div className="openui-preview-empty" role="status">
            Could not render this OpenUI
            {errors[0] ? `: ${errors[0]}` : ""}
          </div>
        )}
        <Renderer
          library={openuiLibrary}
          response={response}
          isStreaming={false}
          onParseResult={(result) => {
            const root =
              result && typeof result === "object"
                ? ((result as { root?: { props?: { children?: unknown } } }).root ?? null)
                : null;
            const kids = root?.props?.children;
            const hasKids = Array.isArray(kids) ? kids.length > 0 : Boolean(kids);
            setParseOk(Boolean(root) && hasKids);
          }}
          onError={(errs) => {
            const list = Array.isArray(errs) ? errs : [];
            setErrors(
              list
                .map((e) =>
                  e && typeof e === "object"
                    ? String((e as { message?: string }).message || e)
                    : String(e)
                )
                .filter(Boolean)
                .slice(0, 3)
            );
            if (list.length) setParseOk(false);
          }}
        />
        {parseOk && (
          <span className="openui-preview-sr" aria-hidden="true">
            rendered
          </span>
        )}
      </div>
    </ThemeProvider>
  );
}

const roots = new WeakMap<Element, Root>();

function mount(el: Element, options: MountOptions) {
  let root = roots.get(el);
  if (!root) {
    root = createRoot(el);
    roots.set(el, root);
  }
  root.render(
    <StrictMode>
      <PreviewApp {...options} />
    </StrictMode>
  );
}

function unmount(el: Element) {
  const root = roots.get(el);
  if (root) {
    root.unmount();
    roots.delete(el);
  }
}

declare global {
  interface Window {
    OpenUIPreview: {
      mount: typeof mount;
      unmount: typeof unmount;
      fillPlaceholders: typeof fillPlaceholders;
    };
  }
}

window.OpenUIPreview = { mount, unmount, fillPlaceholders };
