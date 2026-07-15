import { createContext, useContext } from "react";
import type { JobDef } from "./components";

export interface Caps {
  execution: boolean;
  read_only: boolean;
  jobs_concurrency?: number;
  jobs: JobDef[];
  run_insights?: { browser: boolean; openai_available: boolean };
}

export const CapsContext = createContext<Caps>({
  execution: false,
  read_only: true,
  jobs: [],
});

export const useCaps = () => useContext(CapsContext);

export function jobDef(caps: Caps, name: string): JobDef | undefined {
  return caps.jobs.find((j) => j.job === name);
}
