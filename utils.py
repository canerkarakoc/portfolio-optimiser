from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st


def parse_tickers(text: str) -> List[str]:
    """Parse a comma-separated ticker string into a clean list."""
    tickers = [t.strip().upper() for t in text.split(",") if t.strip()]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in tickers:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result


def export_dataframe(
    df: pd.DataFrame,
    file_basename: str,
    as_excel: bool = False,
    button_label: str = "Export CSV",
) -> None:
    """Render a Streamlit download button for a DataFrame as CSV or Excel."""
    if df.empty:
        st.warning("No data to export.")
        return

    if as_excel:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="data")
        st.download_button(
            label=button_label,
            data=bio.getvalue(),
            file_name=f"{file_basename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label=button_label,
            data=csv,
            file_name=f"{file_basename}.csv",
            mime="text/csv",
        )


