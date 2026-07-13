/**
 * Minimal OpenUI component library for placeholder layout skeletons.
 * Built with official @openuidev/lang-core defineComponent / createLibrary.
 */
import { z } from "zod/v4";
import { defineComponent, createLibrary } from "@openuidev/lang-core";

const Stack = defineComponent({
  name: "Stack",
  description: "Layout container that stacks children vertically or horizontally",
  props: z.object({
    children: z.array(z.any()).describe("Child elements"),
    direction: z
      .enum(["vertical", "horizontal"])
      .optional()
      .describe("Stack direction"),
    gap: z.number().optional().describe("Gap between children"),
  }),
  component: null,
});

const Card = defineComponent({
  name: "Card",
  description: "Card with title and optional body text",
  props: z.object({
    title: z.string().describe("Card title (use placeholder string e.g. :hero.title)"),
    body: z
      .string()
      .optional()
      .describe("Card body (use placeholder string e.g. :hero.body)"),
  }),
  component: null,
});

const Text = defineComponent({
  name: "Text",
  description: "Text block",
  props: z.object({
    content: z
      .string()
      .describe("Text content (use placeholder string e.g. :page.blurb)"),
  }),
  component: null,
});

const Button = defineComponent({
  name: "Button",
  description: "Clickable button",
  props: z.object({
    label: z
      .string()
      .describe("Button label (use placeholder string e.g. :cta.label)"),
  }),
  component: null,
});

export const library = createLibrary({
  root: "Stack",
  id: "slm-training-minimal",
  components: [Stack, Card, Text, Button],
  componentGroups: [
    {
      name: "layout",
      components: ["Stack", "Card"],
      notes: ["Prefer Stack as root; nest Cards and Text inside."],
    },
    {
      name: "content",
      components: ["Text", "Button"],
      notes: [
        "Content props must be placeholder strings like :hero.title — never free-form copy.",
      ],
    },
  ],
});

/** Props that must contain OpenUI placeholder tokens for our training policy. */
export const CONTENT_PROPS = new Set(["title", "body", "content", "label"]);

export const PLACEHOLDER_RE = /^:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/;
