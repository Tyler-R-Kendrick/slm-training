---
version: alpha
name: OpenUI Layout Skeleton
description: Structure-first design context for placeholder-augmented OpenUI scaffolds. Colors are theme context only — not scaffold gold.
colors:
  primary: "#0B57D0"
  on-primary: "#FFFFFF"
  secondary: "#5F6368"
  tertiary: "#C5221F"
  neutral: "#F8F9FA"
  on-surface: "#1A1A1A"
  surface: "#FFFFFF"
typography:
  title:
    fontFamily: Manrope
    fontSize: 1.5rem
    fontWeight: 600
    lineHeight: 1.25
  body:
    fontFamily: Manrope
    fontSize: 1rem
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: Manrope
    fontSize: 0.875rem
    fontWeight: 500
    lineHeight: 1.3
rounded:
  sm: 4px
  md: 8px
  lg: 16px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.md}"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.lg}"
  canvas:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.on-surface}"
  chrome:
    backgroundColor: "{colors.secondary}"
    textColor: "{colors.on-primary}"
  danger:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.on-primary}"
---

## Overview

Structure-first product UI for placeholder OpenUI **layout skeletons**.
Eval and ship gates score **component hierarchy + placeholders only** —
not colors, typography weights, or gap tokens. DESIGN.md may condition the
context tower, but style here is never scaffold gold.

Prefer clear hierarchy and one primary CTA per section. Tone is professional
and calm — suitable for dashboards, forms, and marketing sections without
decorative noise.

## Colors

Theme tokens for optional rendering / context (not OpenUI scaffold args):

- **Primary (#0B57D0):** Interactive actions and key emphasis.
- **On-primary (#FFFFFF):** Text and icons on primary fills.
- **Secondary (#5F6368):** Supporting chrome / metadata.
- **Tertiary (#C5221F):** Destructive highlights only.
- **Neutral / Surface:** Canvas (`neutral`) with dark on-surface body text.

## Typography

Manrope for titles, body, and labels. Scaffold OpenUI must not encode size
enums (`large-heavy`, etc.) — leave type scale to this document / renderer.

## Layout

Prefer `column` Stacks on mobile and `row` for paired actions or metric cards.
Omit OpenUI gap style tokens from gold scaffolds; spacing lives here.
Keep one primary CTA per section.

## Components

- **Buttons:** Structural CTAs with placeholder labels only.
- **Cards:** Surface containers holding TextContent children — never free-form copy.
- **Inputs:** Name + placeholder slots; no styled chrome in OpenUI gold.
- **Images:** ImageBlock with meaningful alt placeholders.

## Do's and Don'ts

- Do: keep user-facing strings as placeholders (`:hero.title`) until a copy model fills them.
- Do: evaluate structure (components, nesting, direction, placeholders) — not color or gap.
- Don't: put hex colors, typography sizes, or gap tokens into OpenUI gold.
- Don't: treat DESIGN.md lint warnings (unused style tokens) as model or ship failures.
