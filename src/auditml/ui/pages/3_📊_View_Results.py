"""AuditML — View Results page."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Results — AuditML",
    page_icon="📊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .metric-good { color: #22c55e; font-weight: 700; }
    .metric-warn { color: #f59e0b; font-weight: 700; }
    .metric-bad  { color: #ef4444; font-weight: 700; }
    .section-head {
        border-left: 4px solid #667eea;
        background: linear-gradient(90deg, #667eea0d, transparent);
        padding: 0.4rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        font-weight: 800;
        font-size: 1.05rem;
        color: #1e293b;
        margin: 1.75rem 0 0.9rem 0;
    }
</style>
""", unsafe_allow_html=True)


def _sec(title: str) -> None:
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)


def _colour_metric(val: float) -> str:
    if val >= 0.75:
        return f'<span class="metric-bad">{val:.4f}</span>'
    if val >= 0.60:
        return f'<span class="metric-warn">{val:.4f}</span>'
    return f'<span class="metric-good">{val:.4f}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Results")
    st.markdown("Load any AuditML report directory to explore metrics and visualisations.")
    st.markdown("---")
    st.markdown("""
**Colour guide (attack accuracy):**
- 🟢 < 0.60 — low leakage
- 🟡 0.60–0.75 — moderate leakage
- 🔴 > 0.75 — high leakage

**After exploring**, use the ⬇️ Download section to get the full report ZIP.
""")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 Audit Results")
st.markdown(
    "Enter the path to an AuditML report directory (created by `auditml audit`) "
    "to explore interactive metrics, charts, and DP comparisons."
)

# ── Load report ───────────────────────────────────────────────────────────────
_sec("📁 Load Report")

default = st.session_state.get("last_report_dir", "./outputs/audit")
c_path, c_btn = st.columns([4, 1])
with c_path:
    report_dir_str = st.text_input("Report directory path", value=default, label_visibility="collapsed")
with c_btn:
    load_clicked = st.button("📂 Load", type="primary", use_container_width=True)

if load_clicked:
    rp = Path(report_dir_str)
    if not rp.exists():
        st.error(f"❌ Directory not found: `{rp}`")
        st.stop()
    sf = rp / "audit_summary.json"
    if not sf.exists():
        st.error(f"❌ No `audit_summary.json` in `{rp}`. Is this an AuditML report directory?")
        st.stop()
    with open(sf, encoding="utf-8") as f:
        st.session_state["report_summary"] = json.load(f)
    st.session_state["report_path"] = str(rp)
    st.success(f"✅ Report loaded from `{rp}`")

if "report_summary" not in st.session_state:
    st.info(
        "Enter the path to your report directory and click **Load**.\n\n"
        "After running `auditml audit`, the report is saved to "
        "`<output_dir>/<experiment_name>/` (e.g. `./outputs/my_audit/`)."
    )
    st.stop()

summary: dict = st.session_state["report_summary"]
report_path = Path(st.session_state["report_path"])

# ── Header summary ────────────────────────────────────────────────────────────
st.markdown("---")
h1, h2, h3, h4 = st.columns(4)
ts = summary.get("timestamp", "")
with h1:
    st.metric("Experiment", summary.get("experiment_name", "—"))
with h2:
    st.metric("Generated", ts[:16].replace("T", " ") if ts else "—")
with h3:
    st.metric("Attacks Run", summary.get("num_attacks", 0))
with h4:
    st.metric("DP Comparison", "✅ Yes" if summary.get("dp_enabled") else "❌ No")

# ── Attack metrics ────────────────────────────────────────────────────────────
_sec("🎯 Attack Results")

attack_metrics: dict = summary.get("attack_metrics", {})
if attack_metrics:
    rows = []
    for name, m in attack_metrics.items():
        rows.append({
            "Attack": name,
            "Accuracy": round(m.get("accuracy", 0.0), 4),
            "AUC-ROC": round(m.get("auc_roc", 0.0), 4),
            "F1 Score": round(m.get("f1", 0.0), 4),
            "Precision": round(m.get("precision", 0.0), 4),
            "Recall": round(m.get("recall", 0.0), 4),
            "TPR@1%FPR": round(m.get("tpr_at_1fpr", 0.0), 4),
            "TPR@0.1%FPR": round(m.get("tpr_at_01fpr", 0.0), 4),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy", min_value=0, max_value=1, format="%.4f",
            ),
            "AUC-ROC": st.column_config.ProgressColumn(
                "AUC-ROC", min_value=0, max_value=1, format="%.4f",
            ),
        },
    )

    best = max(attack_metrics.items(), key=lambda x: x[1].get("auc_roc", 0))
    worst = min(attack_metrics.items(), key=lambda x: x[1].get("auc_roc", 0))
    b_acc = best[1].get("accuracy", 0)
    col_b, col_w = st.columns(2)
    with col_b:
        colour = "metric-bad" if b_acc > 0.75 else "metric-warn" if b_acc > 0.6 else "metric-good"
        st.markdown(
            f'🏆 **Most effective:** {best[0]} '
            f'(AUC-ROC = <span class="{colour}">{best[1].get("auc_roc", 0):.4f}</span>)',
            unsafe_allow_html=True,
        )
    with col_w:
        st.markdown(
            f'🛡️ **Weakest attack:** {worst[0]} '
            f'(AUC-ROC = {worst[1].get("auc_roc", 0):.4f})'
        )
else:
    st.info("No attack metrics found in this report.")

# ── Model accuracy ────────────────────────────────────────────────────────────
model_acc: dict = summary.get("model_accuracy", {})
if model_acc:
    _sec("🧠 Model Utility")
    cols = st.columns(len(model_acc))
    for col, (name, acc) in zip(cols, model_acc.items()):
        with col:
            delta = None
            if name == "dp" and "baseline" in model_acc:
                delta = f"{acc - model_acc['baseline']:+.4f} vs baseline"
            st.metric(
                f"{name.title()} Model",
                f"{acc:.4f}",
                delta=delta,
                delta_color="inverse",
            )

# ── Visualisations ────────────────────────────────────────────────────────────
_sec("📈 Visualisations")

comp_dir = report_path / "attack_comparison"
if comp_dir.exists():
    st.markdown("#### ⚔️ Attack Comparison")
    bar_png = comp_dir / "attack_comparison_bar.png"
    roc_png = comp_dir / "attack_roc_overlay.png"
    if bar_png.exists() or roc_png.exists():
        img_cols = st.columns(2)
        with img_cols[0]:
            if bar_png.exists():
                st.image(str(bar_png), caption="Metrics Bar Chart — All Attacks", use_column_width=True)
        with img_cols[1]:
            if roc_png.exists():
                st.image(str(roc_png), caption="ROC Curve Overlay — All Attacks", use_column_width=True)

attacks_dir = report_path / "attacks"
if attacks_dir.exists():
    attack_subdirs = sorted(d for d in attacks_dir.iterdir() if d.is_dir())
    if attack_subdirs:
        st.markdown("#### 🔍 Per-Attack Details")
        tabs = st.tabs([d.name for d in attack_subdirs])
        for tab, atk_dir in zip(tabs, attack_subdirs):
            with tab:
                mf = atk_dir / "metrics.json"
                if mf.exists():
                    with open(mf, encoding="utf-8") as f:
                        m = json.load(f)
                    cols = st.columns(min(len(m), 4))
                    for col, (k, v) in zip(cols, list(m.items())[:4]):
                        with col:
                            st.metric(
                                k.replace("_", " ").title(),
                                f"{v:.4f}" if isinstance(v, float) else str(v),
                            )
                pngs = sorted(atk_dir.glob("*.png"))
                if pngs:
                    img_c = st.columns(min(len(pngs), 3))
                    for ic, png in zip(img_c, pngs):
                        with ic:
                            st.image(str(png), caption=png.stem.replace("_", " ").title(),
                                     use_column_width=True)

# ── DP comparison ─────────────────────────────────────────────────────────────
if summary.get("dp_enabled"):
    _sec("🛡️ Differential Privacy Comparison")

    if summary.get("epsilon") is not None:
        st.info(
            f"**Privacy Budget ε = {summary['epsilon']:.2f}** — "
            "Smaller ε = stronger privacy but more accuracy loss."
        )

    dp_comp = summary.get("dp_comparison")
    if dp_comp:
        rows2 = []
        for name, gain in dp_comp.items():
            acc_red = gain.get("attack_accuracy_reduction", 0.0)
            rows2.append({
                "Attack": name,
                "Baseline Acc Reduction": round(acc_red, 4),
                "AUC-ROC Reduction": round(gain.get("auc_roc_reduction", 0.0), 4),
                "Utility Cost": round(gain.get("utility_cost", 0.0), 4),
                "Privacy Gain": "✅ Good" if acc_red > 0.05 else "⚠️ Marginal",
            })
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    dp_dir = report_path / "dp_comparison"
    if dp_dir.exists():
        dp_subdirs = sorted(d for d in dp_dir.iterdir() if d.is_dir())
        if dp_subdirs:
            dp_tabs = st.tabs([d.name for d in dp_subdirs])
            for tab, dp_atk_dir in zip(dp_tabs, dp_subdirs):
                with tab:
                    pngs = sorted(dp_atk_dir.glob("*.png"))
                    if pngs:
                        img_c = st.columns(min(len(pngs), 2))
                        for ic, png in zip(img_c, pngs):
                            with ic:
                                st.image(str(png),
                                         caption=png.stem.replace("_", " ").title(),
                                         use_column_width=True)

# ── Text summary ──────────────────────────────────────────────────────────────
_sec("📄 Summary Report")
txt_file = report_path / "summary.txt"
if txt_file.exists():
    with st.expander("View full text summary"):
        st.code(txt_file.read_text(encoding="utf-8"), language="text")
else:
    st.info("No `summary.txt` found in this report directory.")

# ── Raw JSON ──────────────────────────────────────────────────────────────────
with st.expander("🔍 Raw audit_summary.json"):
    st.json(summary)

# ── Download ──────────────────────────────────────────────────────────────────
_sec("⬇️ Download")
st.markdown("Download the entire report (all JSON, text, and PNG files) as a ZIP archive.")

if st.button("📦 Package report as ZIP"):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(report_path.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(report_path))
    zip_buf.seek(0)
    exp = summary.get("experiment_name", "audit")
    st.download_button(
        label="⬇️ Download report.zip",
        data=zip_buf,
        file_name=f"{exp}_report.zip",
        mime="application/zip",
        type="primary",
    )
