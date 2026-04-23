"""
Rust Belt Revival Labor Archive - Scraper

Approach (as required):
  Step 1. Use BeautifulSoup to scrape the main INDEX page and extract
          the href of every individual brief.
  Step 2. Loop over those URLs one at a time. For each page, use
          BeautifulSoup to find its table and build a pandas DataFrame
          from it. Append each DataFrame to a list.
  Step 3. After the loop, combine the list of DataFrames with pd.concat
          into one big panel, clean the messy values, and save.
"""

import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

INDEX_URL = "https://bana290-assignment3.netlify.app/"
HEADERS = {"User-Agent": "Mozilla/5.0 (BANA290 assignment scraper)"}


# ------------------------------------------------------------------
# Step 1. Scrape the index page for brief URLs
# ------------------------------------------------------------------
def get_brief_urls(index_url: str) -> list[str]:
    """Parse the index page with BeautifulSoup and return absolute
    URLs for every individual brief linked under /briefs/."""
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/briefs/" in href:
            full = urljoin(index_url, href)
            if full not in urls:                    # dedupe
                urls.append(full)
    return urls


# ------------------------------------------------------------------
# Step 2a. Helper: turn one HTML <table> into a DataFrame using
#          BeautifulSoup directly (no pd.read_html).
# ------------------------------------------------------------------
def table_to_dataframe(table_tag) -> pd.DataFrame:
    """Walk a <table> with BeautifulSoup and build a DataFrame.

    Works whether or not the table uses <thead>/<tbody>.
    """
    all_rows = table_tag.find_all("tr")
    if not all_rows:
        return pd.DataFrame()

    # First row = headers
    header_cells = all_rows[0].find_all(["th", "td"])
    headers = [c.get_text(" ", strip=True) for c in header_cells]

    # Remaining rows = data
    rows = []
    for tr in all_rows[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells and any(c != "" for c in cells):
            rows.append(cells)

    df = pd.DataFrame(rows, columns=headers)
    # Normalize header whitespace and drop escape noise like "STATE\_GROUP"
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\\", "")).strip()
                  for c in df.columns]
    return df


def canonicalize_brief(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the logical columns of one brief to canonical names so
    multiple briefs with different header styles concat cleanly."""
    col_region   = find_col([r"region", r"county.*name", r"^county$"],
                            df.columns, required=True)
    col_state    = find_col([r"state.*group", r"^state$", r"district"], df.columns)
    col_program  = find_col([r"program.*status", r"status", r"designation",
                             r"treatment", r"grant"], df.columns)
    col_industry = find_col([r"anchor.*industry", r"industry.*anchor",
                             r"industry", r"sector"], df.columns)
    col_note     = find_col([r"portal.*note", r"note", r"memo", r"comment"],
                            df.columns)

    rename = {col_region: "region"}
    if col_state:    rename[col_state]    = "state_group"
    if col_program:  rename[col_program]  = "program_status"
    if col_industry: rename[col_industry] = "anchor_industry"
    if col_note:     rename[col_note]     = "portal_note"
    return df.rename(columns=rename)


# ------------------------------------------------------------------
# Step 2b. Scrape one brief page -> raw DataFrame (+ provenance)
# ------------------------------------------------------------------
def scrape_one_brief(url: str) -> pd.DataFrame:
    """Visit one brief URL and return its table as a DataFrame,
    with columns renamed to canonical names and tagged with source."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    h1 = soup.find("h1")
    brief_title = h1.get_text(strip=True) if h1 else url

    table = soup.find("table")
    if table is None:
        raise RuntimeError(f"No <table> found on {url}")

    df = table_to_dataframe(table)
    df = canonicalize_brief(df)           # <<< normalize headers per brief
    df["brief_title"] = brief_title
    df["brief_url"] = url
    return df


# ------------------------------------------------------------------
# Step 3 helpers. Clean messy employment values and county names.
# ------------------------------------------------------------------
def clean_employment(raw):
    """Parse messy employment strings into an integer number of jobs.

    Handles: '32,620', '32,055 jobs', '~30.9k', '31.4 thousand',
             '34.6 K', '15.2k', '1,234', unicode minus/dashes, and
             common missing-value markers ('', '-', '--', '\u2014',
             'n/a', 'na', 'nan', 'null'). Returns pandas NA on failure.
    """
    if raw is None:
        return pd.NA
    s = str(raw).strip().lower()
    # Unicode dashes -> ascii
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    # Missing markers
    if s in {"", "-", "--", "n/a", "na", "nan", "null", "none"}:
        return pd.NA
    # Drop noise tokens
    s = s.replace("jobs", "").replace("~", "").replace("approx", "")
    s = s.replace(",", "").strip()
    # Detect multiplier
    multiplier = 1.0
    if "thousand" in s:
        multiplier = 1000.0
        s = s.replace("thousand", "")
    elif re.search(r"\bk\b|k$", s):
        multiplier = 1000.0
        s = re.sub(r"k", "", s)
    elif "million" in s or re.search(r"\bm\b|m$", s):
        multiplier = 1_000_000.0
        s = re.sub(r"million|m", "", s)
    s = s.strip()
    try:
        return int(round(float(s) * multiplier))
    except ValueError:
        return pd.NA


def clean_county_name(raw):
    """Standardize county names to 'X County, SS' form.

    Handles: 'Lucas Cnty, Ohio', 'Stark County / OH',
             'Mahoning County - Ohio', 'Erie County (PA)',
             'Beaver Cnty., PA', 'Lawrence Co., PA',
             plus bold+descriptor glued text like 'Lucas Cnty, Ohio Lake Erie Production Belt'.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return pd.NA
    s = str(raw).strip()
    # If BS4 glued the county name and a descriptor line, keep only the line
    # up to and including the state; drop anything after.
    s = s.split("\n")[0]
    # Normalize county-abbreviation variants: Cnty, Cnty., Co., Co  -> County
    s = re.sub(r"\bCnty\b\.?", "County", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCo\b\.?(?=[\s,])", "County", s, flags=re.IGNORECASE)
    # Normalize separators between name and state
    s = s.replace(" / ", ", ").replace(" - ", ", ")
    # "(PA)" -> ", PA"
    s = re.sub(r"\s*\(([^)]+)\)", r", \1", s)
    # Full state names -> abbreviations (extend here for other states if needed)
    state_map = {
        r"\bOhio\b": "OH", r"\bPennsylvania\b": "PA",
        r"\bMichigan\b": "MI", r"\bIndiana\b": "IN",
        r"\bNew York\b": "NY", r"\bWest Virginia\b": "WV",
    }
    for pat, abbr in state_map.items():
        s = re.sub(pat, abbr, s, flags=re.IGNORECASE)
    # Collapse whitespace and tidy commas
    s = re.sub(r"\s+", " ", s).strip().rstrip(",")
    s = re.sub(r"\s*,\s*", ", ", s)
    # If the string contains a County + state abbrev, return just that slice
    m = re.search(r"([A-Z][A-Za-z\.\-\s]+?County),\s*([A-Z]{2})\b", s)
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"
    # Else try a plain "Name, XX" (no "County" word)
    m = re.search(r"([A-Z][A-Za-z\.\-\s]+?),\s*([A-Z]{2})\b", s)
    if m:
        return f"{m.group(1).strip()} County, {m.group(2)}"
    return s


def find_col(patterns, columns, required=False):
    """Match a logical column by regex against actual header names."""
    for pat in patterns:
        for c in columns:
            if re.search(pat, str(c), re.IGNORECASE):
                return c
    if required:
        raise RuntimeError(f"Could not find any of {patterns} in {list(columns)}")
    return None


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main():
    # ---- Step 1: get all brief URLs from the index ----
    print(f"[1] Fetching index: {INDEX_URL}")
    brief_urls = get_brief_urls(INDEX_URL)
    print(f"    found {len(brief_urls)} brief URLs:")
    for u in brief_urls:
        print(f"      - {u}")

    # ---- Step 2: loop through each URL, scrape its table ----
    frames = []                                      # list of DataFrames
    for i, url in enumerate(brief_urls, start=1):
        print(f"\n[2.{i}] Scraping {url}")
        df = scrape_one_brief(url)
        print(f"      shape = {df.shape}, columns = {list(df.columns)[:5]}...")
        frames.append(df)
        time.sleep(0.3)                              # be polite

    # ---- Step 3: concat and clean ----
    raw = pd.concat(frames, ignore_index=True)
    print(f"\n[3] Combined raw table: {raw.shape}")

    # Year columns (4-digit) vs identifier columns
    year_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}", str(c))]
    id_cols   = [c for c in raw.columns if c not in year_cols]

    # Wide -> long
    long_df = raw.melt(id_vars=id_cols, value_vars=year_cols,
                       var_name="year", value_name="employment_raw")

    long_df["year"] = long_df["year"].astype(int)
    long_df["employment"] = long_df["employment_raw"].apply(clean_employment)
    long_df["county"] = long_df["region"].apply(clean_county_name)

    # ---- TREATED: derive from the brief URL, which is the authoritative
    # source (the index page labels Ohio corridors as treated and PA
    # benchmarks as control). Cross-check against program_status text.
    def treated_from_url(url: str) -> int:
        u = str(url).lower()
        if "ohio" in u or "corridor" in u or "grant" in u:
            return 1
        if "benchmark" in u or "pennsylvania" in u or "control" in u:
            return 0
        return pd.NA  # unknown -> flagged below

    long_df["treated"] = long_df["brief_url"].apply(treated_from_url)

    if "program_status" in long_df.columns:
        def status_says_treated(s):
            s = str(s).lower()
            if any(k in s for k in ("grant", "ai ", " ai", "funded", "treated")):
                return 1
            if any(k in s for k in ("benchmark", "control")):
                return 0
            return pd.NA
        status_flag = long_df["program_status"].apply(status_says_treated)
        # NA-safe comparison: convert both to nullable Int64, then build a
        # plain boolean mask by treating any NA result as "no disagreement".
        url_int    = long_df["treated"].astype("Int64")
        status_int = pd.array(status_flag.tolist(), dtype="Int64")
        disagree = (
            url_int.notna() & pd.Series(status_int).notna()
            & (url_int.fillna(-1) != pd.Series(status_int).fillna(-1))
        )
        if bool(disagree.any()):
            print(f"\n[!] Warning: {int(disagree.sum())} rows where URL-based "
                  f"treated != status-based treated. Using URL-based value.")

    long_df["treated"] = long_df["treated"].astype("Int64")

    # ---- POST_POLICY: 2022 grant launch ----
    long_df["post_policy"] = (long_df["year"] >= 2022).astype("Int64")
    # Interaction: if treated is NA, the interaction is NA (not silently 0)
    long_df["treated_x_post"] = (long_df["treated"] * long_df["post_policy"]).astype("Int64")

    # Nullable integer dtype for employment (preserves NA)
    long_df["employment"] = long_df["employment"].astype("Int64")

    keep = [c for c in [
        "county", "state_group", "program_status", "anchor_industry",
        "year", "employment_raw", "employment",
        "treated", "post_policy", "treated_x_post",
        "portal_note", "brief_title", "brief_url",
    ] if c in long_df.columns]
    panel = long_df[keep].copy()

    # ---- Final cleaning pass ----
    # Strip whitespace from every string column
    for c in panel.select_dtypes(include="object").columns:
        panel[c] = panel[c].astype(str).str.strip()

    # Drop rows with an unusable county key
    panel = panel[panel["county"].notna() & (panel["county"] != "nan")]

    # De-duplicate on county-year (keep first occurrence)
    before = len(panel)
    panel = panel.drop_duplicates(subset=["county", "year"], keep="first")
    after = len(panel)
    if before != after:
        print(f"[!] Dropped {before - after} duplicate county-year rows.")

    panel = panel.sort_values(["county", "year"]).reset_index(drop=True)

    # ---- Panel completeness check ----
    counts = panel.groupby("county")["year"].nunique()
    expected = panel["year"].nunique()
    bad = counts[counts != expected]
    if not bad.empty:
        print(f"[!] Incomplete panel: {len(bad)} counties missing years:")
        print(bad.to_string())

    # Save
    panel.to_csv("labor_panel_long.csv", index=False)
    panel.pivot_table(
        index=["county", "state_group", "program_status", "anchor_industry", "treated"],
        columns="year", values="employment",
    ).reset_index().to_csv("labor_panel_wide.csv", index=False)

    # Summary
    print("\n=== Summary ===")
    print(f"rows (long)      : {len(panel)}")
    print(f"unique counties  : {panel['county'].nunique()}")
    print(f"treated counties : {panel.loc[panel['treated']==1,'county'].nunique()}")
    print(f"control counties : {panel.loc[panel['treated']==0,'county'].nunique()}")
    print(f"year range       : {panel['year'].min()}\u2013{panel['year'].max()}")
    print("\nFirst 8 rows:")
    print(panel.head(8).to_string(index=False))
    return panel


if __name__ == "__main__":
    main()
