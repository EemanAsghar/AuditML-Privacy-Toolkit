# About AuditML

## Project

AuditML is a final-year project (FYP) developed at NUML Faisalabad. It provides an end-to-end privacy auditing toolkit for PyTorch models, covering:

- Four privacy attacks (Threshold MIA, Shadow MIA, Model Inversion, Attribute Inference)
- Differential Privacy training via Opacus
- Automated report generation with metrics and visualisations
- Self-contained HTML reports with inline charts (auto-opens in browser)
- An optional Rust acceleration module (~11× faster threshold scanning)

## Author

**Eeman Asghar**  
NUML Faisalabad, 2024–2025

## References

| Paper | Relevance |
|---|---|
| Yeom et al., *Privacy Risk in Machine Learning*, IEEE CSF 2018 | Threshold MIA |
| Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P 2017 | Shadow MIA |
| Fredrikson et al., *Model Inversion Attacks*, CCS 2015 | Model Inversion |
| Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016 | DP-SGD |
| Carlini et al., *Membership Inference Attacks from First Principles*, IEEE S&P 2022 | LiRA and TPR@low FPR |

## Tech stack

- [PyTorch](https://pytorch.org/) — model training and inference
- [Opacus](https://opacus.ai/) — Differential Privacy
- [scikit-learn](https://scikit-learn.org/) — metrics
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) — documentation
- [Rust / PyO3 / maturin](https://www.maturin.rs/) — performance extension

## License

MIT License. See `LICENSE` in the repository root.

## Source

[github.com/EemanAsghar/AuditML](https://github.com/EemanAsghar/AuditML)
