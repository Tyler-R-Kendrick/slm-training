// Metric-lever helpers shared by compiled pages and the interpreted
// toolProvider. The server derives its metric surfaces from the ship-gate
// policy (see slm_training/web/observability.py gate_metric_keys); these
// helpers keep client-side labels and the smoke canary gate on the same
// policy so lever changes propagate everywhere.
export const METRIC_LABELS: Record<string, string> = {
  meaningful_program_rate: "Meaningful v1",
  structural_similarity: "Gold structure",
  component_type_recall: "Gold type recall",
  placeholder_fidelity: "Slot fidelity",
  reward_score: "Reward",
};

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key.replace(/_/g, " ");
}

/** The smoke canary's headline lever + threshold from a /api/gates/policy payload. */
export function smokeGate(policy: Record<string, Record<string, number>> | undefined): {
  lever: string;
  threshold?: number;
  label: string;
} {
  const smoke = policy?.smoke ?? {};
  const lever = Object.keys(smoke)[0] ?? "meaningful_program_rate";
  const threshold = typeof smoke[lever] === "number" ? smoke[lever] : undefined;
  const label =
    threshold === undefined ? "gate" : `≥${threshold} ${metricLabel(lever).toLowerCase()}`;
  return { lever, threshold, label };
}
