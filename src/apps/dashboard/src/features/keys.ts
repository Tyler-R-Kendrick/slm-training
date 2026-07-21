export const DASHBOARD_DEFAULT_RENDERER = "dashboard.default-renderer";
export const VSS_DECODE_ENABLED = "vss.decode-enabled";
export const PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT =
  "playground.grammar-constrained-default";

export const PRODUCT_FLAG_KEYS = [
  DASHBOARD_DEFAULT_RENDERER,
  VSS_DECODE_ENABLED,
  PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT,
] as const;

export type ProductFlagKey = (typeof PRODUCT_FLAG_KEYS)[number];

export type DashboardRenderer = "compiled" | "interpreted";
