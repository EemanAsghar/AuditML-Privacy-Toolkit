"""AuditML — Run Experiment page."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import streamlit as st

st.set_page_config(
    page_title="Run Experiment — AuditML",
    page_icon="▶️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .log-box {
        background: #0f172a;
        color: #e2e8f0;
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        border: 1px solid #334155;
    }
    .status-ok  { color: #22c55e; font-weight: 700; }
    .status-err { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ▶️ Run Experiment")
    st.markdown("Provide a YAML config then click **Run**.")
    st.markdown("---")
    st.markdown("""
**Commands:**
- `auditml audit` — full privacy audit
- `auditml train` — training only

**After running**, go to **📊 Results** to explore the report.
""")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("▶️ Run Experiment")
st.markdown(
    "Provide a YAML configuration, choose a command, and execute the "
    "AuditML pipeline. Output is captured and displayed below."
)

# ── Config source ──────────────────────────────────────────────────────────────
st.markdown("### 📁 Configuration")
src = st.radio(
    "Config source",
    ["From Configure page", "Upload YAML file", "Paste YAML text"],
    horizontal=True,
)

config_yaml: str | None = None

if src == "From Configure page":
    if "config_yaml" in st.session_state:
        config_yaml = st.session_state["config_yaml"]
        cfg_preview = st.session_state.get("config_dict", {})
        st.success("✅ Using config from ⚙️ Configure page.")
        with st.expander("Preview config"):
            st.code(config_yaml, language="yaml")
    else:
        st.warning(
            "⚠️ No config found from Configure page. "
            "Go to **⚙️ Configure** first, or choose another source."
        )

elif src == "Upload YAML file":
    uploaded = st.file_uploader("Upload config.yaml", type=["yaml", "yml"])
    if uploaded:
        config_yaml = uploaded.read().decode("utf-8")
        st.success("✅ File loaded.")
        with st.expander("Preview config"):
            st.code(config_yaml, language="yaml")

elif src == "Paste YAML text":
    pasted = st.text_area(
        "Paste YAML config",
        height=220,
        placeholder="experiment_name: my_audit\ndata:\n  dataset: cifar10\n...",
    )
    if pasted.strip():
        config_yaml = pasted

# ── Command ───────────────────────────────────────────────────────────────────
st.markdown("### 🎛️ Command")
col_cmd, col_desc = st.columns([1, 2])
with col_cmd:
    command = st.radio(
        "Select command",
        ["audit", "train"],
        format_func=lambda x: {
            "audit": "🔍 audit — full privacy audit (recommended)",
            "train": "🏋️ train — training only",
        }[x],
    )
with col_desc:
    if command == "audit":
        st.info(
            "**`auditml audit`** runs the complete pipeline:\n"
            "1. Load or train the baseline model\n"
            "2. Execute all configured attacks\n"
            "3. (Optional) Train DP model and re-run attacks\n"
            "4. Generate unified report"
        )
    else:
        st.info(
            "**`auditml train`** trains only:\n"
            "1. Load dataset and split member/non-member\n"
            "2. Train model (standard or with DP)\n"
            "3. Evaluate accuracy and save checkpoint"
        )

# ── Run ───────────────────────────────────────────────────────────────────────
st.markdown("### 🚀 Execute")

if not config_yaml:
    st.warning("⬆️ Provide a configuration above before running.")
    st.stop()

try:
    parsed = yaml.safe_load(config_yaml)
    exp_name = parsed.get("experiment_name", "audit")
    out_root = parsed.get("reporting", {}).get("output_dir", "./outputs")
    report_path = Path(out_root) / exp_name
except Exception:
    exp_name = "audit"
    report_path = Path("./outputs/audit")

col_run, col_info2 = st.columns([1, 2])
with col_run:
    run_clicked = st.button(
        f"▶️  Run  `auditml {command}`",
        type="primary",
        use_container_width=True,
    )
with col_info2:
    st.markdown(
        f"**Experiment:** `{exp_name}`  \n"
        f"**Output:** `{report_path}`"
    )

if run_clicked:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(config_yaml)
        tmp_path = tmp.name

    st.markdown("#### 📟 Live Output")
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    status_placeholder.info("⏳ Starting…")

    full_log = ""
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "auditml.cli", command, "--config", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            full_log += line
            log_placeholder.markdown(
                f'<div class="log-box">{full_log}</div>',
                unsafe_allow_html=True,
            )
        process.wait()

        if process.returncode == 0:
            status_placeholder.success("✅ Completed successfully!")
            st.session_state["last_report_dir"] = str(report_path)

            if report_path.exists():
                st.success(
                    f"📁 Report saved to `{report_path}`  \n"
                    "Navigate to **📊 View Results** to explore it."
                )
            else:
                st.info(
                    "Run complete. If the report directory was configured differently, "
                    "enter its path manually in **📊 View Results**."
                )
        else:
            status_placeholder.error(
                f"❌ Command exited with code {process.returncode}. "
                "Check the output above for details."
            )

    except FileNotFoundError:
        status_placeholder.error(
            "❌ `auditml` not found. Make sure the package is installed:\n"
            "```bash\npip install -e .\n```"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # ── Log download ──────────────────────────────────────────────────────────
    if full_log:
        st.download_button(
            label="⬇️ Download run log",
            data=full_log,
            file_name=f"{exp_name}_{command}.log",
            mime="text/plain",
        )
