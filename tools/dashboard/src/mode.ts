import { createContext, useContext } from "react";

// "compiled" = hand-written React pages; "interpreted" = live-render each page's
// OpenUI DSL. Mirrors the theme toggle (localStorage + a data-* attribute on :root).
export type RenderMode = "compiled" | "interpreted";
export const ModeContext = createContext<RenderMode>("compiled");
export const useMode = () => useContext(ModeContext);
