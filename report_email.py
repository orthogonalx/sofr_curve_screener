# =============================================================================
# report_email.py  — Email prediction / summary tables (SMTP optional)
# =============================================================================

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional, Sequence

import pandas as pd

from curve_config import (
    EMAIL_ENABLED,
    REPORT_FROM,
    REPORT_TO,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
)


def _cfg(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def format_summary_text(summary: pd.DataFrame, asof: Optional[str] = None) -> str:
    """Plain-text body: as-of + rounded results table."""
    show = summary.copy()
    col_rz = "residual/zscore"
    for c in ("nivel", "predicted", col_rz):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    # z-score rows: predicted is unused — show blank, not nan
    if "predicted" in show.columns:
        show["predicted"] = show["predicted"].apply(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        )
    if col_rz in show.columns:
        show[col_rz] = show[col_rz].apply(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        )
    if "nivel" in show.columns:
        show["nivel"] = show["nivel"].apply(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        )
    cols = [
        c
        for c in ["model", "structure", "nivel", "predicted", col_rz, "equation", "time"]
        if c in show.columns
    ]
    show = show[cols]
    header = "SOFR screener — results"
    if asof:
        header += f"  |  asof {asof}"
    return header + "\n\n" + show.to_string(index=False) + "\n"


def send_summary_email(
    summary: pd.DataFrame,
    asof: Optional[str] = None,
    subject: Optional[str] = None,
    to: Optional[Sequence[str]] = None,
    body: Optional[str] = None,
) -> bool:
    """
    Send summary. Returns True if sent, False if skipped (disabled / incomplete).
    Credentials from curve_config or env (SMTP_* / REPORT_TO / REPORT_FROM).
    Pass body= to override format_summary_text (e.g. prediction tables).
    """
    enabled = os.environ.get("EMAIL_ENABLED", str(EMAIL_ENABLED)).lower() in {
        "1", "true", "yes",
    }
    host = _cfg("SMTP_HOST", SMTP_HOST)
    port = int(_cfg("SMTP_PORT", str(SMTP_PORT)) or 587)
    user = _cfg("SMTP_USER", SMTP_USER)
    password = _cfg("SMTP_PASSWORD", SMTP_PASSWORD)
    mail_from = _cfg("REPORT_FROM", REPORT_FROM) or user
    mail_to: List[str] = list(to) if to else list(REPORT_TO)
    env_to = _cfg("REPORT_TO")
    if env_to:
        mail_to = [x.strip() for x in env_to.split(",") if x.strip()]

    text = body if body is not None else format_summary_text(summary, asof=asof)
    subj = subject or f"SOFR screener {asof or ''}".strip()

    if not enabled or not host or not mail_to or not mail_from:
        if body is None:
            print(text, flush=True)
        print(
            "[email skipped] set EMAIL_ENABLED=true and SMTP_HOST / REPORT_TO "
            "(and credentials) to send",
            flush=True,
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subj
    msg["From"] = mail_from
    msg["To"] = ", ".join(mail_to)
    msg.attach(MIMEText(text, "plain", "utf-8"))

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.sendmail(mail_from, mail_to, msg.as_string())
    return True
