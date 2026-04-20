"""AuditML — Experiment Configuration page."""

from __future__ import annotations

import yaml
import streamlit as st

st.set_page_config(
    page_title="Configure — AuditML",
    page_icon="⚙️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .sec {
        border-left: 4px solid #667eea;
        background: linear-gradient(90deg, #667eea0d, transparent);
        padding: 0.5rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        font-weight: 800;
        font-size: 1.05rem;
        color: #1e293b;
        margin: 1.5rem 0 0.8rem 0;
    }
    .attack-pill {
        display: inline-block;
        background: #667eea18;
        border: 1px solid #667eea44;
        color: #4f46e5;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configure")
    st.markdown("Build your experiment config without writing YAML.")
    st.markdown("---")
    st.markdown("After configuring, either:\n- ⬇️ Download the YAML\n- ▶️ Go to Run Experiment")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("⚙️ Experiment Configuration")
st.markdown(
    "Build your experiment visually. The YAML preview updates live — "
    "download it or carry it directly to **▶️ Run Experiment**."
)

col_form, col_preview = st.columns([1, 1], gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT — Form
# ═══════════════════════════════════════════════════════════════════════════════
with col_form:

    experiment_name = st.text_input(
        "Experiment Name", value="my_audit",
        help="Used as the output sub-directory name.",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📁 Dataset</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dataset = st.selectbox(
            "Dataset",
            ["cifar10", "mnist", "cifar100"],
            help="CIFAR-10 (10 classes) · MNIST (10 digits) · CIFAR-100 (100 classes)",
        )
    with c2:
        train_ratio = st.slider(
            "Member Ratio", 0.1, 0.9, 0.5, 0.05,
            help="Fraction of training data treated as 'members' for attack evaluation.",
        )
    c3, c4 = st.columns([2, 1])
    with c3:
        data_dir = st.text_input("Data Directory", value="./data")
    with c4:
        download = st.checkbox("Auto-download", value=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🧠 Model Architecture</div>', unsafe_allow_html=True)
    arch = st.selectbox(
        "Architecture",
        ["cnn", "resnet"],
        format_func=lambda x: {
            "cnn": "SimpleCNN — fast, lightweight, 3 conv layers",
            "resnet": "SmallResNet — deeper, GroupNorm (required for DP)",
        }[x],
    )
    if arch == "resnet":
        st.info("SmallResNet uses GroupNorm and is compatible with Opacus DP training.")
    else:
        st.info("SimpleCNN is fast and works well on MNIST and CIFAR-10.")

    # ── Attacks ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">⚔️ Privacy Attacks</div>', unsafe_allow_html=True)

    ATTACK_LABELS = {
        "mia_threshold":     "🎯 Threshold MIA — fast baseline",
        "mia_shadow":        "👥 Shadow MIA — trains shadow models, most powerful",
        "model_inversion":   "🖼️ Model Inversion — gradient pixel reconstruction",
        "attribute_inference": "🔍 Attribute Inference — sensitive attribute prediction",
    }
    attacks = st.multiselect(
        "Select Attacks to Run",
        options=list(ATTACK_LABELS.keys()),
        default=["mia_threshold"],
        format_func=lambda x: ATTACK_LABELS[x],
    )
    if not attacks:
        st.warning("⚠️ Select at least one attack.")

    # ── Training ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🏋️ Training</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        epochs = st.number_input("Epochs", 1, 300, 30)
    with c6:
        batch_size = st.number_input("Batch Size", 8, 512, 64, step=8)
    with c7:
        lr = st.number_input("Learning Rate", 1e-5, 0.1, 1e-3, format="%.5f")

    c8, c9 = st.columns(2)
    with c8:
        optimizer = st.selectbox("Optimizer", ["adam", "sgd"])
    with c9:
        device = st.selectbox("Device", ["auto", "cpu", "cuda"],
                              help="auto = use CUDA if available, else CPU")

    c10, c11 = st.columns(2)
    with c10:
        weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 1e-4, format="%.5f")
    with c11:
        seed = st.number_input("Random Seed", 0, 99999, 42)

    # ── Differential Privacy ──────────────────────────────────────────────────
    st.markdown('<div class="sec">🛡️ Differential Privacy</div>', unsafe_allow_html=True)
    dp_enabled = st.toggle(
        "Enable DP Training",
        value=False,
        help="Train a DP model alongside the baseline and compare attack results.",
    )
    epsilon, delta, max_grad_norm, noise_multiplier = 5.0, 1e-5, 1.0, None

    if dp_enabled:
        ca, cb = st.columns(2)
        with ca:
            epsilon = st.number_input(
                "Privacy Budget (ε)", 0.1, 100.0, 5.0,
                help="Lower = stronger privacy but more accuracy loss.",
            )
        with cb:
            delta = st.number_input(
                "Delta (δ)", 1e-7, 1e-3, 1e-5, format="%.2e",
                help="Should be less than 1/(training set size).",
            )
        cc, cd = st.columns(2)
        with cc:
            max_grad_norm = st.number_input("Max Grad Norm (C)", 0.1, 10.0, 1.0)
        with cd:
            auto_noise = st.checkbox("Auto-calibrate noise", value=True,
                                     help="Opacus will set σ to achieve target ε.")
            if not auto_noise:
                noise_multiplier = st.number_input("Noise Multiplier (σ)", 0.1, 10.0, 1.1)

        st.success(f"DP training will target **ε = {epsilon:.2f}** with **δ = {delta:.1e}**.")
    else:
        auto_noise = True

    # ── Reporting ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📊 Output</div>', unsafe_allow_html=True)
    output_dir = st.text_input("Output Directory", value="./outputs")

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT — Preview + Actions
# ═══════════════════════════════════════════════════════════════════════════════
with col_preview:
    st.markdown("### 📋 Live YAML Preview")

    # Build config dict
    cfg: dict = {
        "experiment_name": experiment_name,
        "data": {
            "dataset": dataset,
            "train_ratio": float(train_ratio),
            "data_dir": data_dir,
            "download": bool(download),
        },
        "model": {"arch": arch},
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "optimizer": optimizer,
            "weight_decay": float(weight_decay),
            "device": device,
            "seed": int(seed),
        },
        "attacks": attacks,
        "dp": {
            "enabled": bool(dp_enabled),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "max_grad_norm": float(max_grad_norm),
        },
        "reporting": {"output_dir": output_dir},
    }
    if dp_enabled and not auto_noise and noise_multiplier is not None:
        cfg["dp"]["noise_multiplier"] = float(noise_multiplier)

    yaml_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False, indent=2)
    st.code(yaml_str, language="yaml")

    # ── Action buttons ────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download config.yaml",
        data=yaml_str,
        file_name=f"{experiment_name}_config.yaml",
        mime="text/yaml",
        use_container_width=True,
    )

    if st.button("▶️ Save & Go to Run Experiment", type="primary", use_container_width=True):
        st.session_state["config_yaml"] = yaml_str
        st.session_state["config_dict"] = cfg
        st.success("✅ Config saved to session. Navigate to **▶️ Run Experiment** in the sidebar.")

    # ── Selected attacks summary ──────────────────────────────────────────────
    if attacks:
        st.markdown("---")
        st.markdown("**Selected attacks:**")
        pills_html = "".join(
            f'<span class="attack-pill">{ATTACK_LABELS[a].split("—")[0].strip()}</span>'
            for a in attacks
        )
        st.markdown(pills_html, unsafe_allow_html=True)

        st.markdown("")
        ATTACK_DETAIL = {
            "mia_threshold": (
                "**Threshold MIA** — Classifies a sample as a member if the model's "
                "predicted probability for the true label exceeds an optimal threshold τ*. "
                "Runs in seconds. Good as a fast baseline."
            ),
            "mia_shadow": (
                "**Shadow MIA** — Trains multiple shadow models to mimic the target. "
                "Generates (output, membership) training pairs for an attack MLP classifier. "
                "Most powerful attack. Takes several minutes."
            ),
            "model_inversion": (
                "**Model Inversion** — Reconstructs training images by minimising the "
                "cross-entropy loss of a candidate image w.r.t. a target class via "
                "gradient descent. Exposes visual memorisation."
            ),
            "attribute_inference": (
                "**Attribute Inference** — Maps class labels to sensitive groups, trains "
                "an attack MLP on softmax outputs to predict group membership. "
                "Evaluates per-group accuracy."
            ),
        }
        for a in attacks:
            with st.expander(ATTACK_LABELS[a].split("—")[0].strip()):
                st.markdown(ATTACK_DETAIL[a])

    # ── Estimated runtime ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**⏱️ Estimated runtime (CPU):**")
    time_estimates = {
        "mia_threshold": 0.5,
        "mia_shadow": 15.0,
        "model_inversion": 5.0,
        "attribute_inference": 3.0,
    }
    train_time = max(1.0, epochs * 0.5)
    attack_time = sum(time_estimates.get(a, 2.0) for a in attacks)
    dp_time = train_time * 1.5 if dp_enabled else 0
    total = train_time + attack_time + dp_time

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.metric("Training", f"~{train_time:.0f} min")
        st.metric("Attacks", f"~{attack_time:.0f} min")
    with c_t2:
        if dp_enabled:
            st.metric("DP Training", f"~{dp_time:.0f} min")
        st.metric("Total (est.)", f"~{total:.0f} min")
    st.caption("Estimates for CPU. GPU is typically 5–10× faster.")
