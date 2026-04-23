import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

PANEL_CSV = "labor_panel_long.csv"
POLICY_YEAR = 2022          # actual policy start
PLACEBO_YEAR = 2020         # fake policy date used for anticipation check


# ---------------------------------------------------------------
# Load & prep
# ---------------------------------------------------------------
def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["employment"]).copy()
    df["employment"] = df["employment"].astype(float)
    df["log_emp"]    = np.log(df["employment"])
    df["year"]       = df["year"].astype(int)
    df["treated"]    = df["treated"].astype(int)
    df["post_policy"] = (df["year"] >= POLICY_YEAR).astype(int)
    df["treated_x_post"] = df["treated"] * df["post_policy"]
    return df


# ---------------------------------------------------------------
# 1. Visual analysis: group means over time
# ---------------------------------------------------------------
def plot_trends(df: pd.DataFrame, policy_year: int = POLICY_YEAR,
                outfile: str = "fig_trends.png"):
    g = df.groupby(["year", "treated"])["employment"].mean().unstack("treated")
    g.columns = ["Control (PA benchmarks)", "Treated (OH grant corridor)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    g.plot(ax=ax, marker="o", linewidth=2)
    ax.axvline(policy_year - 0.5, linestyle="--", color="gray",
               label=f"Policy ({policy_year})")
    ax.set_title("Mean manufacturing-related employment by group")
    ax.set_xlabel("Year")
    ax.set_ylabel("Employment (mean across counties)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[1] saved {outfile}")
    return g


# ---------------------------------------------------------------
# 2. Parallel-trends assumption test
# ---------------------------------------------------------------
def parallel_trends_test(df: pd.DataFrame, policy_year: int = POLICY_YEAR):
    """Regress log-employment on treated*year within the pre-period.
    A non-significant slope on the interaction supports parallel trends.
    """
    pre = df[df["year"] < policy_year].copy()
    # Center year so the main effect of 'treated' is still interpretable
    pre["year_c"] = pre["year"] - pre["year"].min()

    model = smf.ols("log_emp ~ treated * year_c", data=pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre["county"]})

    interaction_coef = model.params.get("treated:year_c", np.nan)
    interaction_p    = model.pvalues.get("treated:year_c", np.nan)
    verdict = ("parallel trends hold (not significant)"
               if interaction_p > 0.10 else
               "parallel trends VIOLATED (significant pre-trend)")
    print("\n[2] Parallel-trends test (pre-period OLS, clustered SE)")
    print(f"    treated:year_c coefficient = {interaction_coef:.4f}")
    print(f"    p-value                   = {interaction_p:.4f}")
    print(f"    verdict                   = {verdict}")
    return model


# ---------------------------------------------------------------
# 3. Placebo test: pretend policy happened in a pre-period year
# ---------------------------------------------------------------
def placebo_test(df: pd.DataFrame, placebo_year: int = PLACEBO_YEAR,
                 policy_year: int = POLICY_YEAR):
    """Restrict sample to pre-policy years, assign a fake post dummy,
    and estimate the DiD. A significant estimate would signal anticipation
    effects or non-parallel trends (both are bad)."""
    pre = df[df["year"] < policy_year].copy()
    pre["fake_post"] = (pre["year"] >= placebo_year).astype(int)
    pre["treated_x_fake"] = pre["treated"] * pre["fake_post"]

    model = smf.ols(
        "log_emp ~ treated_x_fake + C(county) + C(year)",
        data=pre,
    ).fit(cov_type="cluster", cov_kwds={"groups": pre["county"]})

    coef = model.params["treated_x_fake"]
    pval = model.pvalues["treated_x_fake"]
    verdict = ("no anticipation (placebo insignificant)"
               if pval > 0.10 else
               "ANTICIPATION or pre-trend issue (placebo significant)")
    print(f"\n[3] Placebo DiD with fake policy year = {placebo_year}")
    print(f"    treated_x_fake coefficient = {coef:.4f}")
    print(f"    p-value                   = {pval:.4f}")
    print(f"    verdict                   = {verdict}")
    return model


# ---------------------------------------------------------------
# 4. DiD estimation with two-way fixed effects
# ---------------------------------------------------------------
def did_regression(df: pd.DataFrame):
    """Two-way (county + year) fixed-effects DiD on log-employment,
    standard errors clustered at the county level."""
    model = smf.ols(
        "log_emp ~ treated_x_post + C(county) + C(year)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["county"]})

    coef = model.params["treated_x_post"]
    pct  = (np.exp(coef) - 1) * 100
    ci_lo, ci_hi = model.conf_int().loc["treated_x_post"]
    pval = model.pvalues["treated_x_post"]

    print("\n[4] DiD estimation (TWFE, clustered SE at county)")
    print(f"    treated_x_post coefficient = {coef:.4f}")
    print(f"    interpretation             = ~{pct:+.2f}% effect on employment")
    print(f"    95% CI                     = [{coef:.4f}, {ci_hi:.4f}]"
          f" -> [{(np.exp(ci_lo)-1)*100:+.2f}%, {(np.exp(ci_hi)-1)*100:+.2f}%]")
    print(f"    p-value                    = {pval:.4f}")
    print(f"    N = {int(model.nobs)}, counties = {df['county'].nunique()}, "
          f"years = {df['year'].nunique()}")
    return model


# ---------------------------------------------------------------
# Event study plot: year-by-year treatment effects
# ---------------------------------------------------------------
def event_study(df: pd.DataFrame, policy_year: int = POLICY_YEAR,
                outfile: str = "fig_event_study.png"):
    """Interact treated with year dummies, omitting the year before
    policy as the reference category. Plot the coefficients."""
    df = df.copy()
    df["year"] = df["year"].astype(int)
    ref_year = policy_year - 1

    # Build dummy interactions manually for all years except the reference
    years = sorted(df["year"].unique())
    for y in years:
        if y == ref_year:
            continue
        df[f"T_{y}"] = df["treated"] * (df["year"] == y).astype(int)

    interaction_terms = [f"T_{y}" for y in years if y != ref_year]
    formula = "log_emp ~ " + " + ".join(interaction_terms) + " + C(county) + C(year)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["county"]})

    # Collect coefficients and 95% CI
    rows = []
    for y in years:
        if y == ref_year:
            rows.append({"year": y, "coef": 0.0, "lo": 0.0, "hi": 0.0})
            continue
        term = f"T_{y}"
        coef = model.params[term]
        lo, hi = model.conf_int().loc[term]
        rows.append({"year": y, "coef": coef, "lo": lo, "hi": hi})
    es = pd.DataFrame(rows).sort_values("year")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(es["year"], es["coef"],
                yerr=[es["coef"] - es["lo"], es["hi"] - es["coef"]],
                fmt="o-", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(policy_year - 0.5, linestyle="--", color="gray",
               label=f"Policy ({policy_year})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Effect on log(employment) vs. 2021")
    ax.set_title("Event study: year-by-year treatment effect")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"    saved {outfile}")
    return es


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------
def main():
    df = load_panel(PANEL_CSV)
    print(f"Loaded {len(df)} county-year rows, "
          f"{df['county'].nunique()} counties, "
          f"years {df['year'].min()}-{df['year'].max()}")

    plot_trends(df)
    parallel_trends_test(df)
    placebo_test(df)
    did_model = did_regression(df)
    event_study(df)

    # Save full regression table
    with open("did_regression_summary.txt", "w") as f:
        f.write(str(did_model.summary()))
    print("\nWrote did_regression_summary.txt")


if __name__ == "__main__":
    main()

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
