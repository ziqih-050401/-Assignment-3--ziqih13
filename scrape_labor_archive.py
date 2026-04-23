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


