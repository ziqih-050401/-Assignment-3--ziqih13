import re
import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

INDEX_URL    = "https://bana290-assignment3.netlify.app/"
HEADERS      = {"User-Agent": "Mozilla/5.0 (BANA290 assignment scraper)"}
PANEL_CSV    = "labor_panel_long.csv"
POLICY_YEAR  = 2022
PLACEBO_YEAR = 2020


# ==================================================================
# SCRAPING
# ==================================================================
def get_brief_urls(index_url: str) -> list[str]:
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        if "/briefs/" in a["href"]:
            full = urljoin(index_url, a["href"])
            if full not in urls:
                urls.append(full)
    return urls


def table_to_dataframe(table_tag) -> pd.DataFrame:
    all_rows = table_tag.find_all("tr")
    if not all_rows:
        return pd.DataFrame()
    header_cells = all_rows[0].find_all(["th", "td"])
    headers = [c.get_text(" ", strip=True) for c in header_cells]
    rows = []
    for tr in all_rows[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells and any(c != "" for c in cells):
            rows.append(cells)
    df = pd.DataFrame(rows, columns=headers)
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\\", "")).strip()
                  for c in df.columns]
    return df


def find_col(patterns, columns, required=False):
    for pat in patterns:
        for c in columns:
            if re.search(pat, str(c), re.IGNORECASE):
                return c
    if required:
        raise RuntimeError(f"Could not find any of {patterns} in {list(columns)}")
    return None


def canonicalize_brief(df: pd.DataFrame) -> pd.DataFrame:
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


def scrape_one_brief(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    h1 = soup.find("h1")
    brief_title = h1.get_text(strip=True) if h1 else url
    table = soup.find("table")
    if table is None:
        raise RuntimeError(f"No <table> found on {url}")
    df = table_to_dataframe(table)
    df = canonicalize_brief(df)
    df["brief_title"] = brief_title
    df["brief_url"] = url
    return df


# ==================================================================
# CLEANING
# ==================================================================
def clean_employment(raw):
    if raw is None:
        return pd.NA
    s = str(raw).strip().lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    if s in {"", "-", "--", "n/a", "na", "nan", "null", "none"}:
        return pd.NA
    s = s.replace("jobs", "").replace("~", "").replace("approx", "")
    s = s.replace(",", "").strip()
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
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return pd.NA
    s = str(raw).strip().split("\n")[0]
    s = re.sub(r"\bCnty\b\.?", "County", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCo\b\.?(?=[\s,])", "County", s, flags=re.IGNORECASE)
    s = s.replace(" / ", ", ").replace(" - ", ", ")
    s = re.sub(r"\s*\(([^)]+)\)", r", \1", s)
    state_map = {
        r"\bOhio\b": "OH", r"\bPennsylvania\b": "PA",
        r"\bMichigan\b": "MI", r"\bIndiana\b": "IN",
        r"\bNew York\b": "NY", r"\bWest Virginia\b": "WV",
    }
    for pat, abbr in state_map.items():
        s = re.sub(pat, abbr, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip().rstrip(",")
    s = re.sub(r"\s*,\s*", ", ", s)
    m = re.search(r"([A-Z][A-Za-z\.\-\s]+?County),\s*([A-Z]{2})\b", s)
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"
    m = re.search(r"([A-Z][A-Za-z\.\-\s]+?),\s*([A-Z]{2})\b", s)
    if m:
        return f"{m.group(1).strip()} County, {m.group(2)}"
    return s


def build_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine scraped brief frames and produce a cleaned county-year panel."""
    raw = pd.concat(frames, ignore_index=True)
    year_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}", str(c))]
    id_cols   = [c for c in raw.columns if c not in year_cols]

    long_df = raw.melt(id_vars=id_cols, value_vars=year_cols,
                       var_name="year", value_name="employment_raw")
    long_df["year"] = long_df["year"].astype(int)
    long_df["employment"] = long_df["employment_raw"].apply(clean_employment)
    long_df["county"]     = long_df["region"].apply(clean_county_name)

    def treated_from_url(url: str):
        u = str(url).lower()
        if "ohio" in u or "corridor" in u or "grant" in u:
            return 1
        if "benchmark" in u or "pennsylvania" in u or "control" in u:
            return 0
        return pd.NA

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
        url_int    = long_df["treated"].astype("Int64")
        status_int = pd.Series(pd.array(status_flag.tolist(), dtype="Int64"))
        disagree = (url_int.notna() & status_int.notna()
                    & (url_int.fillna(-1) != status_int.fillna(-1)))
        if bool(disagree.any()):
            print(f"[!] Warning: {int(disagree.sum())} rows where URL-based "
                  f"treated != status-based treated. Using URL-based value.")

    long_df["treated"]     = long_df["treated"].astype("Int64")
    long_df["post_policy"] = (long_df["year"] >= POLICY_YEAR).astype("Int64")
    long_df["treated_x_post"] = (long_df["treated"] * long_df["post_policy"]).astype("Int64")
    long_df["employment"]  = long_df["employment"].astype("Int64")

    keep = [c for c in [
        "county", "state_group", "program_status", "anchor_industry",
        "year", "employment_raw", "employment",
        "treated", "post_policy", "treated_x_post",
        "portal_note", "brief_title", "brief_url",
    ] if c in long_df.columns]
    panel = long_df[keep].copy()

    for c in panel.select_dtypes(include="object").columns:
        panel[c] = panel[c].astype(str).str.strip()

    panel = panel[panel["county"].notna() & (panel["county"] != "nan")]
    before = len(panel)
    panel = panel.drop_duplicates(subset=["county", "year"], keep="first")
    if len(panel) != before:
        print(f"[!] Dropped {before - len(panel)} duplicate county-year rows.")
    panel = panel.sort_values(["county", "year"]).reset_index(drop=True)

    counts = panel.groupby("county")["year"].nunique()
    bad = counts[counts != panel["year"].nunique()]
    if not bad.empty:
        print(f"[!] Incomplete panel: {len(bad)} counties missing years:\n{bad}")

    return panel


def run_scrape() -> pd.DataFrame:
    print(f"[1] Fetching index: {INDEX_URL}")
    brief_urls = get_brief_urls(INDEX_URL)
    print(f"    found {len(brief_urls)} brief URLs")
    for u in brief_urls:
        print(f"      - {u}")

    frames = []
    for i, url in enumerate(brief_urls, start=1):
        print(f"\n[2.{i}] Scraping {url}")
        df = scrape_one_brief(url)
        print(f"      shape = {df.shape}, columns = {list(df.columns)[:5]}...")
        frames.append(df)
        time.sleep(0.3)

    panel = build_panel(frames)

    panel.to_csv(PANEL_CSV, index=False)
    panel.pivot_table(
        index=["county", "state_group", "program_status", "anchor_industry", "treated"],
        columns="year", values="employment",
    ).reset_index().to_csv("labor_panel_wide.csv", index=False)

    print("\n=== Scrape summary ===")
    print(f"rows (long)      : {len(panel)}")
    print(f"unique counties  : {panel['county'].nunique()}")
    print(f"treated counties : {panel.loc[panel['treated']==1,'county'].nunique()}")
    print(f"control counties : {panel.loc[panel['treated']==0,'county'].nunique()}")
    print(f"year range       : {panel['year'].min()}\u2013{panel['year'].max()}")
    return panel


# ==================================================================
# ANALYSIS
# ==================================================================
def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["employment"]).copy()
    df["employment"]      = df["employment"].astype(float)
    df["log_emp"]         = np.log(df["employment"])
    df["year"]            = df["year"].astype(int)
    df["treated"]         = df["treated"].astype(int)
    df["post_policy"]     = (df["year"] >= POLICY_YEAR).astype(int)
    df["treated_x_post"]  = df["treated"] * df["post_policy"]
    return df


def plot_trends(df, outfile="fig_trends.png"):
    g = df.groupby(["year", "treated"])["employment"].mean().unstack("treated")
    g.columns = ["Control (PA benchmarks)", "Treated (OH grant corridor)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    g.plot(ax=ax, marker="o", linewidth=2)
    ax.axvline(POLICY_YEAR - 0.5, linestyle="--", color="gray",
               label=f"Policy ({POLICY_YEAR})")
    ax.set_title("Mean manufacturing-related employment by group")
    ax.set_xlabel("Year")
    ax.set_ylabel("Employment (mean across counties)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(outfile, dpi=150); plt.close(fig)
    print(f"[A.1] saved {outfile}")


def parallel_trends_test(df):
    pre = df[df["year"] < POLICY_YEAR].copy()
    pre["year_c"] = pre["year"] - pre["year"].min()
    model = smf.ols("log_emp ~ treated * year_c", data=pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre["county"]})
    coef = model.params.get("treated:year_c", np.nan)
    pval = model.pvalues.get("treated:year_c", np.nan)
    verdict = ("parallel trends hold (not significant)" if pval > 0.10
               else "parallel trends VIOLATED (significant pre-trend)")
    print("\n[A.2] Parallel-trends test (pre-period OLS, clustered SE)")
    print(f"      treated:year_c coefficient = {coef:.4f}")
    print(f"      p-value                    = {pval:.4f}")
    print(f"      verdict                    = {verdict}")
    return model


def placebo_test(df):
    pre = df[df["year"] < POLICY_YEAR].copy()
    pre["fake_post"] = (pre["year"] >= PLACEBO_YEAR).astype(int)
    pre["treated_x_fake"] = pre["treated"] * pre["fake_post"]
    model = smf.ols("log_emp ~ treated_x_fake + C(county) + C(year)",
                    data=pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre["county"]})
    coef = model.params["treated_x_fake"]
    pval = model.pvalues["treated_x_fake"]
    verdict = ("no anticipation (placebo insignificant)" if pval > 0.10
               else "ANTICIPATION or pre-trend issue (placebo significant)")
    print(f"\n[A.3] Placebo DiD with fake policy year = {PLACEBO_YEAR}")
    print(f"      treated_x_fake coefficient = {coef:.4f}")
    print(f"      p-value                    = {pval:.4f}")
    print(f"      verdict                    = {verdict}")
    return model


def did_regression(df):
    model = smf.ols("log_emp ~ treated_x_post + C(county) + C(year)",
                    data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["county"]})
    coef = model.params["treated_x_post"]
    pct  = (np.exp(coef) - 1) * 100
    ci_lo, ci_hi = model.conf_int().loc["treated_x_post"]
    pval = model.pvalues["treated_x_post"]
    print("\n[A.4] DiD estimation (TWFE, clustered SE at county)")
    print(f"      treated_x_post coefficient = {coef:.4f}")
    print(f"      interpretation             = ~{pct:+.2f}% effect")
    print(f"      95% CI                     = [{(np.exp(ci_lo)-1)*100:+.2f}%, "
          f"{(np.exp(ci_hi)-1)*100:+.2f}%]")
    print(f"      p-value                    = {pval:.4f}")
    print(f"      N = {int(model.nobs)}, counties = {df['county'].nunique()}")
    return model


def event_study(df, outfile="fig_event_study.png"):
    df = df.copy()
    ref_year = POLICY_YEAR - 1
    years = sorted(df["year"].unique())
    for y in years:
        if y == ref_year:
            continue
        df[f"T_{y}"] = df["treated"] * (df["year"] == y).astype(int)
    terms = [f"T_{y}" for y in years if y != ref_year]
    formula = "log_emp ~ " + " + ".join(terms) + " + C(county) + C(year)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["county"]})

    rows = []
    for y in years:
        if y == ref_year:
            rows.append({"year": y, "coef": 0.0, "lo": 0.0, "hi": 0.0})
        else:
            term = f"T_{y}"
            lo, hi = model.conf_int().loc[term]
            rows.append({"year": y, "coef": model.params[term], "lo": lo, "hi": hi})
    es = pd.DataFrame(rows).sort_values("year")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(es["year"], es["coef"],
                yerr=[es["coef"] - es["lo"], es["hi"] - es["coef"]],
                fmt="o-", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(POLICY_YEAR - 0.5, linestyle="--", color="gray",
               label=f"Policy ({POLICY_YEAR})")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Effect on log(employment) vs. {ref_year}")
    ax.set_title("Event study: year-by-year treatment effect")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(outfile, dpi=150); plt.close(fig)
    print(f"      saved {outfile}")


def run_analysis():
    df = load_panel(PANEL_CSV)
    print(f"\nLoaded {len(df)} county-year rows, "
          f"{df['county'].nunique()} counties, "
          f"years {df['year'].min()}-{df['year'].max()}")
    plot_trends(df)
    parallel_trends_test(df)
    placebo_test(df)
    did_model = did_regression(df)
    event_study(df)
    with open("did_regression_summary.txt", "w") as f:
        f.write(str(did_model.summary()))
    print("\nWrote did_regression_summary.txt")


# ==================================================================
# DRIVER
# ==================================================================
def main():
    run_scrape()
    run_analysis()


if __name__ == "__main__":
    main()




