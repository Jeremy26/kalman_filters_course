"""Visual identity close to thinkautonomous.ai: dark navy to steel blue gradient,
Fira Sans headings, Inter text, blue gradient buttons with 10 px corners.
Fonts come from Google Fonts; offline, the browser falls back to its sans-serif."""

import gradio as gr

BLUE_TOP, BLUE_BOTTOM = "rgb(0, 146, 248)", "rgb(0, 114, 211)"
TEXT = "#e0eef9"
MUTED = "#9fb7cc"
PANEL = "rgba(13, 22, 34, 0.78)"
BORDER = "rgba(120, 170, 205, 0.22)"

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #000000 0%, rgb(31, 30, 75) 45%, rgb(72, 132, 158) 100%)",
    body_background_fill_dark="linear-gradient(135deg, #000000 0%, rgb(31, 30, 75) 45%, rgb(72, 132, 158) 100%)",
    body_text_color=TEXT, body_text_color_dark=TEXT,
    body_text_color_subdued=MUTED, body_text_color_subdued_dark=MUTED,
    block_background_fill=PANEL, block_background_fill_dark=PANEL,
    block_border_color=BORDER, block_border_color_dark=BORDER,
    block_label_text_color=MUTED, block_label_text_color_dark=MUTED,
    block_title_text_color=TEXT, block_title_text_color_dark=TEXT,
    background_fill_primary="rgba(9, 15, 24, 0.9)", background_fill_primary_dark="rgba(9, 15, 24, 0.9)",
    background_fill_secondary="rgba(20, 32, 48, 0.9)",
    background_fill_secondary_dark="rgba(20, 32, 48, 0.9)",
    border_color_primary=BORDER, border_color_primary_dark=BORDER,
    input_background_fill="rgba(6, 12, 20, 0.85)", input_background_fill_dark="rgba(6, 12, 20, 0.85)",
    input_border_color=BORDER, input_border_color_dark=BORDER,
    button_primary_background_fill=f"linear-gradient({BLUE_TOP} 5%, {BLUE_BOTTOM} 100%)",
    button_primary_background_fill_dark=f"linear-gradient({BLUE_TOP} 5%, {BLUE_BOTTOM} 100%)",
    button_primary_background_fill_hover=BLUE_BOTTOM,
    button_primary_background_fill_hover_dark=BLUE_BOTTOM,
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_large_radius="10px", button_small_radius="10px",
    slider_color=BLUE_TOP, slider_color_dark=BLUE_TOP,
    checkbox_background_color_selected=BLUE_TOP, checkbox_background_color_selected_dark=BLUE_TOP,
    checkbox_label_background_fill="rgba(20, 32, 48, 0.9)",
    checkbox_label_background_fill_dark="rgba(20, 32, 48, 0.9)",
    checkbox_label_background_fill_hover="rgba(30, 46, 68, 0.95)",
    checkbox_label_background_fill_hover_dark="rgba(30, 46, 68, 0.95)",
    checkbox_label_background_fill_selected="rgba(0, 146, 248, 0.22)",
    checkbox_label_background_fill_selected_dark="rgba(0, 146, 248, 0.22)",
    checkbox_label_text_color=TEXT, checkbox_label_text_color_dark=TEXT,
    checkbox_label_text_color_selected="#ffffff", checkbox_label_text_color_selected_dark="#ffffff",
    checkbox_label_border_color=BORDER, checkbox_label_border_color_dark=BORDER,
    checkbox_background_color="rgba(6, 12, 20, 0.9)", checkbox_background_color_dark="rgba(6, 12, 20, 0.9)",
    table_even_background_fill="rgba(13, 22, 34, 0.6)", table_even_background_fill_dark="rgba(13, 22, 34, 0.6)",
    table_odd_background_fill="rgba(20, 32, 48, 0.6)", table_odd_background_fill_dark="rgba(20, 32, 48, 0.6)",
    table_border_color=BORDER, table_border_color_dark=BORDER,
    code_background_fill="rgba(0, 146, 248, 0.15)", code_background_fill_dark="rgba(0, 146, 248, 0.15)",
    color_accent=BLUE_TOP, color_accent_soft="rgba(0, 146, 248, 0.25)",
    color_accent_soft_dark="rgba(0, 146, 248, 0.25)",
)

HEAD = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Fira+Sans:wght@500;600;700&family=Inter:wght@400;500;600&display=swap">')

CSS = f"""
.gradio-container {{ max-width: 1680px !important; }}
h1, h2, h3, .ta-title {{ font-family: "Fira Sans", Inter, sans-serif !important; font-weight: 600;
  color: {TEXT} !important; }}
.ta-header {{ display: flex; align-items: center; gap: 14px; padding: 6px 2px 2px; }}
.ta-header svg {{ flex: none; }}
.ta-header h1 {{ margin: 0; font-size: 1.9rem; letter-spacing: 0.2px; }}
.ta-header p {{ margin: 2px 0 0; color: {MUTED}; font-size: 0.95rem; }}
.ta-note {{ color: {MUTED}; font-size: 0.85rem; }}
.ta-metrics table {{ width: 100%; border-collapse: collapse; }}
.ta-metrics th, .ta-metrics td {{ padding: 6px 10px; border-bottom: 1px solid {BORDER}; text-align: left; }}
.ta-metrics th {{ color: {MUTED}; font-weight: 500; }}
code {{ color: {TEXT} !important; background: rgba(0, 146, 248, 0.15) !important; }}
"""

# A plain outlined triangle in the site's blue, not the actual logo file.
HEADER = f"""
<div class="ta-header">
  <svg width="44" height="40" viewBox="0 0 44 40" aria-hidden="true">
    <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{BLUE_TOP}"/><stop offset="1" stop-color="#1f3a8a"/></linearGradient></defs>
    <path d="M22 3 L41 37 L3 37 Z" fill="none" stroke="url(#g)" stroke-width="3" stroke-linejoin="round"/>
  </svg>
  <div><h1>Fusion Lab</h1>
  <p>Kalman filtering and multi-sensor fusion on real nuScenes data</p></div>
</div>
"""
