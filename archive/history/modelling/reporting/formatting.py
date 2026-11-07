"""Shared Markdown table formatting."""
import pandas as pd

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])

def nfmt(x, digits=1):
    return "—" if pd.isna(x) else f"{x:.{digits}f}"
