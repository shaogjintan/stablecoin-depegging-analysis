"""Date-based historical context; annotations are not causal attribution."""
import pandas as pd

def classify(row):
    s = row["start"]
    coin = row["coin"]
    if pd.Timestamp("2023-03-09", tz="UTC") <= s <= pd.Timestamp("2023-03-14", tz="UTC"):
        return "SVB bank run (Mar 2023)"
    if pd.Timestamp("2022-05-07", tz="UTC") <= s <= pd.Timestamp("2022-05-17", tz="UTC"):
        return "LUNA/Terra collapse (May 2022)"
    if pd.Timestamp("2022-11-06", tz="UTC") <= s <= pd.Timestamp("2022-11-14", tz="UTC"):
        return "FTX collapse (Nov 2022)"
    if pd.Timestamp("2022-01-01", tz="UTC") <= s <= pd.Timestamp("2022-05-06", tz="UTC"):
        return "Crypto market stress (early 2022)"
    if s.year == 2021:
        return "DeFi / liquidity stress (2021)"
    # PAX-specific depegs Jan-Mar 2023 (Paxos regulatory pressure, NYDFS BUSD order)
    if pd.Timestamp("2023-01-01", tz="UTC") <= s <= pd.Timestamp("2023-03-08", tz="UTC"):
        return "Paxos regulatory pressure (early 2023)"
    if pd.Timestamp("2023-03-15", tz="UTC") <= s <= pd.Timestamp("2023-12-31", tz="UTC"):
        return "Post-SVB banking contagion (2023)"
    if s.year >= 2024:
        return "Post-regulation stress (2024-25)"
    return "Unclassified (pre-2021 training)"

def annotate(episodes):
    result = episodes.copy()
    result["start"] = pd.to_datetime(result["start"], utc=True)
    result["historical_context"] = result.apply(classify, axis=1)
    return result
