"""AuditML — A privacy auditing toolkit for PyTorch models.

Quick start
-----------
>>> import auditml
>>> results = auditml.audit(model, train_loader, test_loader)
>>> print(results.summary())
"""

from auditml.auditor import AttackSummary, AuditResults, audit, split_loaders

__version__ = "0.1.0"

__all__ = [
    "audit",
    "split_loaders",
    "AuditResults",
    "AttackSummary",
    "__version__",
]
