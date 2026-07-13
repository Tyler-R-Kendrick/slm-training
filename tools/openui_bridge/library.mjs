/**
 * Official OpenUI component library (@openuidev/react-ui openuiLibrary).
 * Placeholder content policy is project-specific (see CONTENT_PROPS).
 */
import { openuiLibrary } from "@openuidev/react-ui/genui-lib";

/** Full openuiLibrary (~54 components, root Stack). */
export const library = openuiLibrary;

/**
 * User-facing string props that must be placeholder tokens for training.
 * Structural/enum props (direction, gap, variant, name keys, etc.) are exempt.
 */
export const CONTENT_PROPS = new Set([
  "text",
  "label",
  "title",
  "body",
  "content",
  "placeholder",
  "alt",
  "hint",
  "description",
  "trigger",
]);

export const PLACEHOLDER_RE =
  /^:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/;

/** Component names for grammar soft priors (derived from library). */
export const COMPONENT_NAMES = Object.keys(library.components || {});
