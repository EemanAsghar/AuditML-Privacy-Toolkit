"""AuditML dashboard landing page."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AuditML — Privacy Auditing Toolkit",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit footer/menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 1.25rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0 0 0.4rem 0;
        letter-spacing: -1px;
    }
    .hero p.sub {
        font-size: 1.35rem;
        opacity: 0.92;
        margin: 0 0 1.5rem 0;
    }
    .hero p.desc {
        font-size: 1.05rem;
        opacity: 0.82;
        max-width: 680px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    .badges {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }

    /* Stats */
    .stat-box {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #667eea18, #764ba218);
        border: 1px solid #667eea33;
        border-radius: 1rem;
    }
    .stat-val {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .stat-lbl {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Feature cards */
    .feat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.75rem 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(102,126,234,0.2);
    }
    .feat-icon { font-size: 3rem; margin-bottom: 0.75rem; }
    .feat-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .feat-desc {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    /* Steps */
    .step {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1.1rem 1.4rem;
        border-radius: 0 0.75rem 0.75rem 0;
        margin-bottom: 1rem;
    }
    .step-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.2rem;
    }
    .step-title { font-size: 1.05rem; font-weight: 700; color: #1e293b; }
    .step-desc  { color: #64748b; font-size: 0.9rem; margin-top: 0.2rem; line-height: 1.5; }

    /* Section heading */
    .sec-head {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e293b;
        margin: 2rem 0 1.2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 2.5rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔒 AuditML")
    st.markdown("Privacy Auditing Toolkit for PyTorch Models")
    st.markdown("---")
    st.markdown("""
**Quick links:**
- ⚙️ Configure — build your experiment
- ▶️ Run — execute the pipeline
- 📊 Results — explore your report

**CLI equivalent:**
```bash
auditml train  --config cfg.yaml
auditml audit  --config cfg.yaml
```
""")
    st.markdown("---")
    st.markdown("📦 `pip install auditml`")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔒 AuditML</h1>
  <p class="sub">Privacy Auditing Toolkit for PyTorch Models</p>
  <p class="desc">
    Measure, compare, and defend against privacy leakage in your machine learning
    models using four state-of-the-art attack methods and built-in
    Differential Privacy training — all from a single dashboard or CLI command.
  </p>
  <div class="badges">
    <span class="badge">⚡ PyTorch 2.x</span>
    <span class="badge">🛡️ Opacus DP</span>
    <span class="badge">🐍 Python 3.10+</span>
    <span class="badge">📊 4 Attack Types</span>
    <span class="badge">🧪 359 Tests Passing</span>
    <span class="badge">📁 YAML Config</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in zip(
    [c1, c2, c3, c4, c5],
    ["4", "3", "2", "359", "ε"],
    ["Privacy Attacks", "Datasets", "Architectures", "Tests Passing", "DP Protection"],
):
    with col:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{val}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">🚀 Four Privacy Attacks, One Tool</div>', unsafe_allow_html=True)

cols = st.columns(4)
features = [
    ("🎯", "Threshold MIA",
     "Detects membership by comparing model confidence to an optimal threshold τ. "
     "Fast, interpretable, and effective as a baseline attack."),
    ("👥", "Shadow MIA",
     "Trains shadow models to mimic the target, then trains an attack MLP classifier. "
     "State-of-the-art membership inference accuracy."),
    ("🖼️", "Model Inversion",
     "Reconstructs approximate training images via gradient-based optimisation. "
     "Directly exposes visual data memorisation."),
    ("🔍", "Attribute Inference",
     "Infers sensitive attributes (demographics, categories) from softmax outputs. "
     "No access to raw data required."),
]
for col, (icon, title, desc) in zip(cols, features):
    with col:
        st.markdown(
            f'<div class="feat-card">'
            f'<div class="feat-icon">{icon}</div>'
            f'<div class="feat-title">{title}</div>'
            f'<div class="feat-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br><br>", unsafe_allow_html=True)

# ── How It Works + Quick Start ────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="sec-head">📋 How It Works</div>', unsafe_allow_html=True)
    for num, title, desc in [
        ("Step 1 — Configure",
         "Choose Dataset & Attacks",
         "Use the visual configuration builder to select your dataset (MNIST/CIFAR-10/CIFAR-100), "
         "model architecture, attack types, training parameters, and DP privacy budget."),
        ("Step 2 — Run",
         "Execute the Pipeline",
         "AuditML trains the model (or loads a checkpoint), runs all selected attacks, "
         "optionally trains a DP-protected model, and re-runs attacks for comparison."),
        ("Step 3 — Results",
         "Explore Your Report",
         "Interactive metrics tables, ROC curves, attack comparison bar charts, and DP vs "
         "non-DP comparisons. Download the full report as a ZIP."),
    ]:
        st.markdown(
            f'<div class="step">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="sec-head">⚡ Quick Start (CLI)</div>', unsafe_allow_html=True)
    st.code("""# Install
pip install auditml

# See all defaults
auditml show-config

# Train a model
auditml train --config config.yaml

# Run full privacy audit
auditml audit --config config.yaml

# Launch this dashboard
auditml ui""", language="bash")

    st.markdown("**Example config.yaml**")
    st.code("""experiment_name: my_audit
data:
  dataset: cifar10
  train_ratio: 0.5
model:
  arch: cnn
training:
  epochs: 30
  batch_size: 64
attacks:
  - mia_threshold
  - mia_shadow
dp:
  enabled: true
  epsilon: 5.0""", language="yaml")

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation CTAs ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">🗺️ Get Started</div>', unsafe_allow_html=True)
n1, n2, n3 = st.columns(3)
with n1:
    st.info("**⚙️ Configure**\n\nBuild your experiment visually — no YAML editing required. "
            "Select attacks, set DP budget, download your config.")
with n2:
    st.success("**▶️ Run Experiment**\n\nUpload a config or use the builder. "
               "Execute `auditml train` or `auditml audit` and watch live output.")
with n3:
    st.warning("**📊 View Results**\n\nLoad any audit report directory. Explore metrics, "
               "ROC curves, comparison charts, and DP analysis. Download as ZIP.")

# ── DP Explainer ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🛡️ What is Differential Privacy (DP)?"):
    dp1, dp2 = st.columns(2)
    with dp1:
        st.markdown("""
**Differential Privacy** provides a mathematical guarantee that the
inclusion or exclusion of any single record in training changes the
model's output distribution by at most a multiplicative factor
controlled by the **privacy budget ε**.

- **Smaller ε** → Stronger privacy, more noise added, lower accuracy
- **Larger ε** → Weaker privacy, less noise, higher accuracy
- **δ** → Small failure probability (typically 1e-5)

AuditML uses **Opacus** to train DP models and automatically compares
attack effectiveness before and after DP — showing you exactly how much
protection you get for the utility cost.
        """)
    with dp2:
        st.code("""# Enable DP in config.yaml
dp:
  enabled: true
  epsilon: 5.0        # privacy budget
  delta: 0.00001      # failure probability
  max_grad_norm: 1.0  # gradient clip norm

# AuditML will:
# 1. Train standard baseline model
# 2. Run attacks → measure leakage
# 3. Train DP model with ε=5.0
# 4. Re-run attacks → measure reduction
# 5. Report privacy gain vs utility cost""", language="yaml")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <strong>AuditML</strong> — Privacy Auditing Toolkit for PyTorch Models<br>
  Final Year Project · Department of Software Engineering · NUML Faisalabad<br>
  <small>Built with Python · PyTorch · Opacus · Streamlit</small>
</div>
""", unsafe_allow_html=True)
