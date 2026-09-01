"""Sunstone North lab styling for Gradio surfaces.

Palette and type are lifted from docs/sidecar_receipts.html, which is already
deployed at lab.sunstonenorth.com/receipts/, so these are the site's own values
rather than an approximation of them.

Gradio 6 takes `theme`, `css` and `head` on launch(), not on Blocks().
"""

from __future__ import annotations

import gradio as gr

# Gradio strips @import from injected CSS, so fonts have to come through head.
LAB_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""

LAB_CSS = """
:root, .gradio-container, gradio-app, body {
  --bg:#faf7f2; --ink:#4a423b; --terra:#b0603e; --muted:#8b8178;
  --line:#e7dfd5; --panel:#ffffff; --sand:#f4eee6;
  --pass:#3d7a44; --fail:#a04533;
}

html, body, gradio-app, .gradio-container {
  background: #faf7f2 !important;
  color: #4a423b !important;
}

:root, .gradio-container {
  --body-background-fill: var(--bg);
  --background-fill-primary: var(--bg);
  --background-fill-secondary: var(--sand);
  --block-background-fill: var(--panel);
  --input-background-fill: var(--panel);
  --input-background-fill-focus: var(--panel);
  --input-background-fill-hover: var(--panel);
  --input-text-color: var(--ink);
  --input-placeholder-color: #b3a99e;
  --input-border-color: var(--line);
  --input-border-color-focus: var(--terra);
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--muted);
  --border-color-primary: var(--line);
  --border-color-accent: var(--terra);
  --block-border-color: var(--line);
  --color-accent: var(--terra);
  --link-text-color: var(--terra);
  --link-text-color-hover: var(--terra);
  --block-label-background-fill: var(--sand);
  --block-label-text-color: var(--muted);
  --block-title-text-color: var(--muted);
  --button-primary-background-fill: var(--terra);
  --button-primary-background-fill-hover: #9c5335;
  --button-primary-text-color: #fff;
  --button-primary-border-color: var(--terra);
  --slider-color: var(--terra);
  --checkbox-background-color-selected: var(--terra);
  --block-radius: 12px;
  --container-radius: 12px;
}

.gradio-container {
  background: var(--bg) !important;
  font: 16px/1.65 Inter, system-ui, sans-serif !important;
  max-width: 1040px !important;
  margin: 0 auto !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: "Playfair Display", serif !important;
  color: var(--ink) !important;
}
.gradio-container h1 { font-size: 42px !important; font-weight: 700 !important; line-height: 1.15 !important; margin: 0 0 10px !important; }
.gradio-container h2 { font-size: 24px !important; font-weight: 600 !important; margin: 28px 0 6px !important; }

.lab-kicker {
  font: 600 11px/1 Inter, sans-serif; letter-spacing: .22em; text-transform: uppercase;
  color: var(--terra); margin-bottom: 14px;
}
.lab-sub { color: var(--muted); font-size: 18px; margin: 0 0 10px; }
.lab-note { color: var(--muted); font-size: 13.5px; border-top: 1px solid var(--line); padding-top: 16px; margin-top: 22px; }
.lab-pill {
  display: inline-block; background: var(--sand); border: 1px solid var(--line);
  border-radius: 999px; padding: 2px 12px; font-size: 12.5px; color: var(--muted);
  margin: 2px 4px 2px 0;
}
.lab-fig { color: var(--terra); font-weight: 600; font-variant-numeric: tabular-nums; }

.gradio-container .block, .gradio-container .form {
  background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important;
}
.gradio-container .panel {
  background: var(--panel) !important; border: 1px solid var(--line) !important;
  border-radius: 12px !important; padding: 4px 16px !important;
}

.gradio-container code, .gradio-container pre {
  font: 13px/1.5 ui-monospace, Menlo, monospace !important;
  background: var(--sand) !important; border-radius: 6px !important;
}
.gradio-container pre { border: 1px solid var(--line) !important; padding: 12px 14px !important; }

.gradio-container table {
  border-collapse: collapse; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; font-size: 14.5px;
}
.gradio-container th { font-weight: 600; background: var(--sand); }
.gradio-container th, .gradio-container td { padding: 9px 12px; border-bottom: 1px solid var(--line); }
.gradio-container td.num, .gradio-container th.num { text-align: right; font-variant-numeric: tabular-nums; }

footer, .gradio-container .icon-button-wrapper { display: none !important; }
.gradio-container .progress-text, .gradio-container .progress-level,
.gradio-container .progress-bar, .gradio-container .meta-text,
.gradio-container .meta-text-center, .gradio-container .eta-bar { display: none !important; }
.gradio-container .pending, .gradio-container .html-container.pending { opacity: 1 !important; }

.gradio-container .label-wrap {
  font: 600 11px/1 Inter, sans-serif !important; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted) !important;
}
.gradio-container .label-wrap:hover { color: var(--terra) !important; }

.gradio-container textarea, .gradio-container input[type="text"] {
  background: var(--panel) !important; color: var(--ink) !important;
  caret-color: var(--terra) !important; border-radius: 10px !important;
}
"""

LAB_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)


def launch_kwargs() -> dict:
    return {"theme": LAB_THEME, "css": LAB_CSS, "head": LAB_HEAD}
