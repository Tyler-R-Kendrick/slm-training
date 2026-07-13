const promptEl = document.getElementById("prompt");
const designMdEl = document.getElementById("design_md");
const generateBtn = document.getElementById("generate");
const grammarEl = document.getElementById("grammar");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const errorEl = document.getElementById("error");
const badgeEl = document.getElementById("badge");
const examplesEl = document.getElementById("examples");

const FALLBACK_EXAMPLES = [
  "Hero card with title and body",
  "Primary call to action button",
  "Two feature cards stacked vertically",
  "Text blurb above a button",
  "Horizontal row of two buttons",
  "Pricing card with subscribe button",
];

function setBusy(busy) {
  generateBtn.disabled = busy;
  generateBtn.textContent = busy ? "Generating…" : "Generate";
  statusEl.textContent = busy ? "Running TwoTower…" : "";
}

function renderExamples(examples) {
  examplesEl.innerHTML = "";
  for (const text of examples) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", () => {
      promptEl.value = text;
      promptEl.focus();
    });
    examplesEl.appendChild(btn);
  }
}

async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const data = await res.json();
    renderExamples(data.examples?.length ? data.examples : FALLBACK_EXAMPLES);
  } catch {
    renderExamples(FALLBACK_EXAMPLES);
  }
}

async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    statusEl.textContent = "Enter a prompt first.";
    promptEl.focus();
    return;
  }

  setBusy(true);
  errorEl.hidden = true;
  badgeEl.textContent = "running";
  badgeEl.className = "badge";

  try {
    const design_md = (designMdEl?.value || "").trim();
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        grammar_constrained: grammarEl.checked,
        design_md: design_md || null,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Generation failed");
    }

    const code = outputEl.querySelector("code") || outputEl;
    code.textContent = data.serialized || data.openui || "";
    // retrigger animation
    outputEl.style.animation = "none";
    void outputEl.offsetWidth;
    outputEl.style.animation = "";

    if (data.valid) {
      badgeEl.textContent = "valid";
      badgeEl.className = "badge ok";
      statusEl.textContent = "Parsed with @openuidev/lang-core";
    } else {
      badgeEl.textContent = "invalid";
      badgeEl.className = "badge bad";
      statusEl.textContent = "Generated, but OpenUI validation failed";
      errorEl.hidden = false;
      errorEl.textContent = data.error || "Validation failed";
    }
  } catch (err) {
    badgeEl.textContent = "error";
    badgeEl.className = "badge bad";
    statusEl.textContent = "";
    errorEl.hidden = false;
    errorEl.textContent = err.message || String(err);
  } finally {
    setBusy(false);
  }
}

generateBtn.addEventListener("click", generate);
promptEl.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    generate();
  }
});

loadExamples();
promptEl.value = FALLBACK_EXAMPLES[0];
