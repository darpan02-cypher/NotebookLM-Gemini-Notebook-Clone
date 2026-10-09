"""Theme + CSS for the Gradio UI. Colors come from CSS variables so light and dark mode both work."""
import gradio as gr

THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f6f7fb",
    body_background_fill_dark="#0e1117",
    block_background_fill="#ffffff",
    block_background_fill_dark="#171b24",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(16,24,40,.04)",
    button_primary_background_fill="linear-gradient(135deg,#6366f1,#8b5cf6)",
    button_primary_background_fill_hover="linear-gradient(135deg,#5558e8,#7c4de6)",
    button_primary_text_color="white",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#0f131b",
)

CSS = """
.gradio-container {
  /* defined here (not :root) so they re-resolve when body.dark redefines the Gradio variables */
  --brand: #6366f1;
  --brand-2: #8b5cf6;
  --soft: rgba(99,102,241,.12);
  --card: var(--block-background-fill);
  --line: var(--border-color-primary);
  --muted: var(--body-text-color-subdued);
}
.gradio-container { max-width: 1480px !important; margin: 0 auto; }
footer { display: none !important; }

/* ---------- hero ---------- */
.hero { display:flex; align-items:center; justify-content:space-between; gap:16px;
        padding: 6px 4px 14px; }
.hero .brand { display:flex; align-items:center; gap:14px; }
.logo { width:44px; height:44px; border-radius:13px; display:grid; place-items:center;
        background: linear-gradient(135deg,var(--brand),var(--brand-2)); color:#fff;
        font-weight:700; font-size:22px; box-shadow:0 6px 18px rgba(99,102,241,.35); }
.hero h1 { margin:0; font-size:22px; font-weight:700; letter-spacing:-.02em; }
.hero p { margin:2px 0 0; font-size:13px; color:var(--muted); }
.theme-btn { border:1px solid var(--line); background:var(--card); color:inherit; cursor:pointer;
             border-radius:999px; padding:8px 14px; font-size:13px; transition:all .15s; }
.theme-btn:hover { border-color:var(--brand); transform:translateY(-1px); }

/* ---------- panels ---------- */
.panel { border:1px solid var(--line); border-radius:18px; padding:16px !important;
         background:var(--card); box-shadow:0 1px 3px rgba(16,24,40,.05); gap:12px !important; }
.panel-title { font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
               color:var(--muted); margin:4px 0 0; }
.panel .block { border:none !important; box-shadow:none !important; background:transparent !important; }
.panel .form { border:none !important; background:transparent !important; box-shadow:none !important; }
.status { font-size:13px; color:var(--muted); min-height:0 !important; margin:0 !important; padding:0 !important; }
.status p { margin:0; }

button { transition: transform .12s ease, box-shadow .12s ease, filter .12s ease !important; }
button.primary:hover { transform:translateY(-1px); box-shadow:0 8px 20px rgba(99,102,241,.35); }
button.secondary:hover, button.stop:hover { transform:translateY(-1px); }

/* ---------- source cards ---------- */
.src-list { display:flex; flex-direction:column; gap:8px; max-height:260px; overflow:auto; }
.src { display:flex; align-items:center; gap:10px; padding:10px 12px; border:1px solid var(--line);
       border-radius:12px; background:var(--card); }
.src .badge { font-size:10px; font-weight:700; padding:4px 7px; border-radius:7px; color:#fff;
              letter-spacing:.04em; flex:none; }
.badge.pdf { background:#ef4444; } .badge.pptx { background:#f59e0b; }
.badge.txt { background:#64748b; } .badge.url { background:#0ea5e9; }
.src .meta { min-width:0; flex:1; }
.src .name { font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.src .sub { font-size:11px; color:var(--muted); }
.pill { font-size:11px; padding:3px 9px; border-radius:999px; font-weight:600; flex:none; }
.pill.ok { background:rgba(16,185,129,.14); color:#059669; }
.pill.bad { background:rgba(239,68,68,.14); color:#dc2626; }
.pill.wait { background:rgba(245,158,11,.16); color:#d97706; }
.empty { text-align:center; color:var(--muted); font-size:13px; padding:18px 8px;
         border:1.5px dashed var(--line); border-radius:12px; }

/* ---------- chat ---------- */
.seg .wrap { flex-wrap:nowrap !important; gap:6px !important; justify-content:flex-end; }
.seg label { white-space:nowrap; }
.chat-head { display:flex; align-items:center; justify-content:space-between; }
.chips button { border-radius:999px !important; font-size:12.5px !important; padding:6px 12px !important;
                background:var(--soft) !important; color:var(--brand) !important; border:1px solid transparent !important; }
.chips button:hover { border-color:var(--brand) !important; }
.ask textarea { font-size:15px !important; }

/* ---------- citations ---------- */
.cites { display:flex; flex-direction:column; gap:8px; }
.cites-head { font-size:12px; color:var(--muted); display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.tag { background:var(--soft); color:var(--brand); padding:2px 9px; border-radius:999px; font-weight:600; }
.cite { display:flex; gap:10px; padding:10px 12px; border:1px solid var(--line); border-radius:12px;
        background:var(--card); }
.cite .n { flex:none; width:22px; height:22px; border-radius:7px; display:grid; place-items:center;
           background:linear-gradient(135deg,var(--brand),var(--brand-2)); color:#fff; font-size:12px; font-weight:700; }
.cite .src-line { font-size:12.5px; font-weight:600; }
.cite .src-line a { color:var(--brand); text-decoration:none; }
.cite .snip { font-size:12px; color:var(--muted); margin-top:3px; line-height:1.45; }

/* ---------- studio ---------- */
.card-btn { text-align:left !important; }
.studio-hint { font-size:12.5px; color:var(--muted); margin:-4px 0 2px; }
.doc-view table { font-size:13px; }
.doc-view { max-height:420px; overflow:auto; padding:14px 16px !important; border:1px solid var(--line) !important;
            border-radius:12px !important; background:var(--card) !important; font-size:14px; line-height:1.6; }


/* =====================================================================
   DUSK: twilight aurora background, frosted-glass panels, warm "paper"
   reading surfaces, amber-coral accent. Layered on top of Gradio dark.
   ===================================================================== */
/* NOTE: Gradio scopes every selector under `.contain`, so `.dusk` lives on a wrapper inside it. */
.dusk {
  --card: rgba(255,255,255,.07); --line: rgba(255,255,255,.15); --muted: #b9aed3;
  --brand: #f6b455; --brand-2: #f472b6; --soft: rgba(246,180,85,.16);
  --body-background-fill: transparent;
  --background-fill-primary: rgba(255,255,255,.05);
  --background-fill-secondary: rgba(255,255,255,.05);
  --block-background-fill: rgba(255,255,255,.06);
  --block-border-color: rgba(255,255,255,.14);
  --border-color-primary: rgba(255,255,255,.14);
  --body-text-color: #f3eefc;
  --body-text-color-subdued: #b9aed3;
  --input-background-fill: rgba(255,255,255,.08);
  --input-border-color: rgba(255,255,255,.16);
  --button-secondary-background-fill: rgba(255,255,255,.11);
  --button-secondary-background-fill-hover: rgba(255,255,255,.18);
  --button-secondary-text-color: #f3eefc;
  --button-primary-background-fill: linear-gradient(135deg,#f6b455,#f472b6);
  --button-primary-background-fill-hover: linear-gradient(135deg,#fbbf63,#f78bc4);
  --button-primary-text-color: #2a1538;
}

.dusk .panel { background: rgba(255,255,255,.07) !important; border-color: rgba(255,255,255,.15);
               backdrop-filter: blur(16px) saturate(140%); -webkit-backdrop-filter: blur(16px) saturate(140%);
               box-shadow: 0 10px 40px rgba(8,4,24,.45), inset 0 1px 0 rgba(255,255,255,.10); }
.dusk .hero h1 { background: linear-gradient(90deg,#fff,#f6d9a8); -webkit-background-clip:text;
                 background-clip:text; color:transparent; }
.dusk .logo { background: linear-gradient(135deg,#f6b455,#f472b6); color:#2a1538;
              box-shadow:0 8px 24px rgba(246,180,85,.4); }
.dusk .theme-btn { background: rgba(255,255,255,.10); color:#f3eefc; border-color: rgba(255,255,255,.2); }
.dusk button.primary { color:#2a1538 !important; font-weight:700; }
.dusk button.primary:hover { box-shadow:0 8px 24px rgba(244,114,182,.4); }

/* inputs, dropdowns, upload zone */
.dusk input, .dusk textarea { color:#f3eefc !important; }
.dusk ::placeholder { color:#a99dc7 !important; }
.dusk .src, .dusk .cite { background: rgba(255,255,255,.07); border-color: rgba(255,255,255,.14); }
.dusk .cite .n { color:#2a1538; }
.dusk .tag { background: rgba(246,180,85,.18); color:#f6c978; }
.dusk .chips button { background: rgba(246,180,85,.14) !important; color:#f6c978 !important; }
.dusk .pill.ok { background: rgba(52,211,153,.18); color:#6ee7b7; }
.dusk .seg label { background: rgba(255,255,255,.12) !important; }
.dusk .seg label, .dusk .seg label span { color:#f3eefc !important; }
.dusk .seg label.selected, .dusk .seg input:checked + span { color:#fff !important; }
.dusk .badge { box-shadow: 0 2px 8px rgba(0,0,0,.25); }

/* chat: answers on warm paper, questions in amber */
.dusk .bot.message, .dusk .bot .message { background:#fbf6ec !important; color:#2b2236 !important;
      border:none !important; box-shadow:0 4px 18px rgba(8,4,24,.35); }
.dusk .bot.message *, .dusk .bot .message * { color:#2b2236 !important; }
.dusk .user.message, .dusk .user .message { background: linear-gradient(135deg,#f6b455,#f59e6b) !important;
      color:#2a1538 !important; border:none !important; box-shadow:0 4px 18px rgba(246,180,85,.3); }
.dusk .user.message *, .dusk .user .message * { color:#2a1538 !important; }
.dusk .bubble-wrap { background: transparent !important; }

/* generated documents on paper */
.dusk .doc-view { background:#fbf6ec !important; border-color:#eadfca !important; color:#2b2236; }
.dusk .doc-view * { color:#2b2236 !important; }
.dusk .doc-view th, .dusk .doc-view td { border-color:#e3d7c0 !important; }

@media (max-width: 900px) {
  .hero h1 { font-size:19px; }
  .panel { padding:12px !important; }
}
"""

# Runs once on page load: defines the Light -> Dark -> Dusk cycle and restores the saved choice.
THEME_JS = """() => {
  const DUSK_BG = 'radial-gradient(1000px 620px at 6% -8%, rgba(124,92,255,.50), transparent 65%), ' +
    'radial-gradient(900px 560px at 100% 2%, rgba(20,184,166,.34), transparent 62%), ' +
    'radial-gradient(700px 500px at 30% 55%, rgba(124,92,255,.14), transparent 70%), ' +
    'radial-gradient(1000px 640px at 52% 112%, rgba(244,114,182,.32), transparent 62%)';
  const modes = ['light', 'dark', 'dusk'];
  const labels = {light: '☀️ Light', dark: '🌙 Dark', dusk: '🌆 Dusk'};
  const apply = (m) => {
    const b = document.body;
    b.classList.remove('dark', 'dusk');
    if (m !== 'light') b.classList.add('dark');
    if (m === 'dusk') b.classList.add('dusk');
    // Gradio scopes custom CSS inside .contain, so mirror the class on a wrapper there
    const wrap = document.querySelector('.gradio-container .contain > *');
    if (wrap) wrap.classList.toggle('dusk', m === 'dusk');
    // Aurora glow: a fixed click-through overlay blended onto the page (independent of Gradio's own backgrounds)
    let glow = document.getElementById('nb-aurora');
    if (m === 'dusk' && !glow) {
      glow = document.createElement('div');
      glow.id = 'nb-aurora';
      glow.style.cssText = 'position:fixed;inset:0;z-index:0;pointer-events:none;mix-blend-mode:screen;background:' + DUSK_BG;
      document.body.appendChild(glow);
    }
    if (glow) glow.style.display = m === 'dusk' ? 'block' : 'none';
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = labels[m];
    try { localStorage.setItem('nb-theme', m); } catch (e) {}
    window.__nbTheme = m;
  };
  window.cycleNbTheme = () => apply(modes[(modes.indexOf(window.__nbTheme || 'light') + 1) % modes.length]);
  const fromUrl = new URLSearchParams(location.search).get('theme');
  let saved = null;
  try { saved = localStorage.getItem('nb-theme'); } catch (e) {}
  const system = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  apply(modes.includes(fromUrl) ? fromUrl : (modes.includes(saved) ? saved : system));
}"""
