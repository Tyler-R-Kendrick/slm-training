// Module-level holder for the app's router navigate fn, set by DslView on each
// render. Custom library components (NavChip, JobList cancel) call navRef.current
// to drive in-app navigation with the exact "chip" styling compiled mode uses,
// without threading React context through the OpenUI Renderer.
export const navRef: { current: ((to: string) => void) | null } = { current: null };
