import {
  OpenFeature,
  TypedInMemoryProvider,
  type EvaluationContext,
  type FlagValue,
  type JsonValue,
} from "@openfeature/web-sdk";
import { getJSON } from "../api";
import {
  DASHBOARD_DEFAULT_RENDERER,
  type DashboardRenderer,
  type ProductFlagKey,
} from "./keys";

export interface FeatureBootstrap {
  provider: "in_memory" | "posthog" | "launchdarkly";
  posthog: { project_api_key: string; host: string } | null;
  launchdarkly: boolean;
  defaults: Record<string, JsonValue>;
  evaluated: Record<string, JsonValue>;
  targeting_key: string;
  flags: Array<{
    key: string;
    kind: string;
    description: string;
    matrix_ref: string | null;
    provider_affinity: string;
  }>;
}

let bootstrapCache: FeatureBootstrap | null = null;
let initPromise: Promise<FeatureBootstrap> | null = null;

function targetingKey(): string {
  const key = "slm-targeting-key";
  let id = localStorage.getItem(key);
  if (!id) {
    id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `anon-${Date.now()}`;
    localStorage.setItem(key, id);
  }
  return id;
}

function toInMemoryFlags(
  evaluated: Record<string, JsonValue>,
): Record<string, { defaultVariant: string; variants: Record<string, FlagValue>; disabled: false }> {
  const flags: Record<
    string,
    { defaultVariant: string; variants: Record<string, FlagValue>; disabled: false }
  > = {};
  for (const [key, value] of Object.entries(evaluated)) {
    flags[key] = {
      defaultVariant: "default",
      variants: { default: value as FlagValue },
      disabled: false,
    };
  }
  return flags;
}

async function initPostHogProvider(bootstrap: FeatureBootstrap): Promise<void> {
  if (!bootstrap.posthog) {
    throw new Error("posthog bootstrap config missing");
  }
  const [{ PostHogWebProvider }, posthog] = await Promise.all([
    import("@posthog/openfeature-web-provider"),
    import("posthog-js"),
  ]);
  posthog.default.init(bootstrap.posthog.project_api_key, {
    api_host: bootstrap.posthog.host,
    loaded: (client) => {
      client.identify(bootstrap.targeting_key);
    },
  });
  await OpenFeature.setProviderAndWait(new PostHogWebProvider(posthog.default));
  await OpenFeature.setContext({ targetingKey: bootstrap.targeting_key });
}

export async function initFeatureRuntime(): Promise<FeatureBootstrap> {
  if (bootstrapCache) return bootstrapCache;
  if (!initPromise) {
    initPromise = (async () => {
      const key = targetingKey();
      const bootstrap = await getJSON<FeatureBootstrap>(
        `/api/features/bootstrap?targeting_key=${encodeURIComponent(key)}`,
      );
      if (bootstrap.provider === "posthog" && bootstrap.posthog) {
        try {
          await initPostHogProvider(bootstrap);
          bootstrapCache = bootstrap;
          return bootstrap;
        } catch (err) {
          console.warn("PostHog OpenFeature provider failed; using bootstrap snapshot", err);
        }
      }
      if (bootstrap.provider === "launchdarkly") {
        // Server-side LD evaluation only — hydrate client from evaluated snapshot.
      }
      await OpenFeature.setProviderAndWait(
        new TypedInMemoryProvider(toInMemoryFlags(bootstrap.evaluated)),
      );
      await OpenFeature.setContext({ targetingKey: bootstrap.targeting_key });
      bootstrapCache = bootstrap;
      return bootstrap;
    })();
  }
  return initPromise;
}

export function getFeatureClient() {
  return OpenFeature.getClient();
}

export function getStringFlag(key: ProductFlagKey, fallback: string): string {
  return getFeatureClient().getStringValue(key, fallback);
}

export function getBooleanFlag(key: ProductFlagKey, fallback: boolean): boolean {
  return getFeatureClient().getBooleanValue(key, fallback);
}

export function trackFeatureExposure(
  event: string,
  attributes?: Record<string, string | number | boolean>,
): void {
  getFeatureClient().track(event, attributes);
}

export function defaultDashboardRenderer(fallback: DashboardRenderer = "interpreted"): DashboardRenderer {
  const value = getStringFlag(DASHBOARD_DEFAULT_RENDERER, fallback);
  return value === "compiled" ? "compiled" : "interpreted";
}

export async function withFeatureContext(
  patch: EvaluationContext,
  fn: () => void | Promise<void>,
): Promise<void> {
  const prior = OpenFeature.getContext();
  await OpenFeature.setContext({ ...prior, ...patch });
  try {
    await fn();
  } finally {
    await OpenFeature.setContext(prior);
  }
}
