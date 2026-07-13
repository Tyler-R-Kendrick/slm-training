---
version: alpha
name: OpenUI Layout Skeleton
description: Default design system for placeholder-augmented OpenUI generation.
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
---

## Overview

Clean product UI for placeholder OpenUI layout skeletons. Prefer clear hierarchy,
generous spacing, and high-contrast primary actions. Tone is professional and calm —
suitable for dashboards, forms, and marketing sections without decorative noise.

## Colors

- **Primary (#0B57D0):** Interactive actions and key emphasis.
- **On-primary (#FFFFFF):** Text and icons on primary fills.
- **Secondary (#5F6368):** Supporting labels, borders, metadata.
- **Tertiary (#C5221F):** Destructive or critical highlights only.
- **Neutral / Surface:** Light canvas with dark on-surface body text.

## Typography

Manrope for titles, body, and labels. Titles are medium-heavy; body is readable at 1rem;
labels stay compact for buttons and form chrome.

## Layout

Prefer `column` Stacks on mobile and `row` for paired actions or metric cards.
Use OpenUI gap tokens (`s`, `m`, `l`) aligned to the spacing scale. Keep one primary CTA
per section.

## Components

- **Buttons:** Filled primary for the main action; secondary for alternatives.
- **Cards:** Surface containers holding TextContent children — never free-form copy.
- **Inputs:** Clear placeholders; labels via adjacent TextContent when needed.
- **Images:** ImageBlock with meaningful alt placeholders.

## Do's and Don'ts

- Do: keep user-facing strings as placeholders (`:hero.title`) until a copy model fills them.
- Do: reference color tokens instead of inventing one-off hex values.
- Don't: overload the first viewport with competing CTAs or dense chrome.
- Don't: sacrifice contrast — keep primary actions WCAG AA against their backgrounds.
