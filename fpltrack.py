"""
Streamlit FPL Assistant (complete)
=================================

This file implements a multi-page Streamlit app that helps with Fantasy Premier League
selection. Features included:
- Player Explorer (filters, sortable table)
- Fixtures view with Fixture Difficulty Rating (FDR)
- Projected Points model (simple, explainable)
- Transfer suggestion engine (single transfer / simple heuristics)
- My Team importer (paste names or IDs), squad validation (100m cap, 3 per team, position counts),
  auto-fix suggestions and an "Apply Auto-Fix" button
- Save / Load squad to disk so squads persist across restarts

Inline comments explain the more complex parts of the logic.

How to run:
1. pip install streamlit requests pandas
2. streamlit run this_file.py

"""

import streamlit as st
import requests
import pandas as pd
import json
import os
from datetime import timedelta
from typing import List, Tuple, Dict, Optional

# --------------------
# Configuration
# --------------------
APP_TITLE = "⚽ FPL Assistant"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
SAVED_SQUAD_FILE = "saved_squad.json"  # file to persist squad between restarts
SQUAD_BUDGET = 100.0  # million
POSITION_RULES = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}  # required counts
MAX_PER_TEAM = 3

# --------------------
# Data fetching with caching
# --------------------
@st.cache_data(ttl=60 * 10)
def fetch_bootstrap() -> dict:
    """Fetch bootstrap-static data from FPL API and return parsed JSON."""
    resp = requests.get(BOOTSTRAP_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=60 * 10)
def fetch_fixtures() -> List[dict]:
    """Fetch fixtures list; return raw JSON list of fixtures."""
    resp = requests.get(FIXTURES_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()

# --------------------
# Data builders
# --------------------
def build_players_df(bootstrap: dict) -> pd.DataFrame:
    """Convert bootstrap 'elements' to a friendly DataFrame with computed metrics.

    We keep team as its numeric id (for merging with fixtures) and also provide a
    human-friendly team_name for display.
    """
    elements = pd.DataFrame(bootstrap.get("elements", []))
    teams = pd.DataFrame(bootstrap.get("teams", []))[ ["id", "name", "short_name"] ]
    types = pd.DataFrame(bootstrap.get("element_types", []))[ ["id", "singular_name_short"] ]

    # merge in team name and position short name
    elements = elements.merge(teams, left_on="team", right_on="id", how="left", suffixes=("", "_team"))
    elements = elements.merge(types, left_on="element_type", right_on="id", how="left", suffixes=("", "_etype"))

    # compute convenient columns
    df = elements.copy()
    df["team_id"] = df["team"]  # numeric id
    df["team_name"] = df["short_name"]
    df["position"] = df["singular_name_short"]
    df["price_m"] = df["now_cost"].astype(float) / 10.0  # FPL stores cost as int (e.g., 55 = 5.5)

    # safe numeric conversions
    df["form"] = pd.to_numeric(df.get("form", 0), errors="coerce").fillna(0.0)
    df["total_points"] = pd.to_numeric(df.get("total_points", 0), errors="coerce").fillna(0.0)
    df["points_per_game"] = pd.to_numeric(df.get("points_per_game", 0), errors="coerce").fillna(0.0)
    df["selected_by_percent"] = pd.to_numeric(df.get("selected_by_percent", 0), errors="coerce").fillna(0.0)

    # value metric: points per million spent
    # avoid division by zero
    df["ppM"] = df.apply(lambda r: (r["total_points"] / r["price_m"]) if r["price_m"] > 0 else 0, axis=1)

    # keep relevant columns
    keep = ["id", "web_name", "first_name", "second_name", "team_id", "team_name", "position", "price_m",
            "total_points", "form", "points_per_game", "selected_by_percent", "ppM", "minutes"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]

# --------------------
# Fixture Difficulty Rating (FDR)
# --------------------
def build_fdr(fixtures: List[dict], horizon_days: int = 42) -> pd.DataFrame:
    """Compute an average difficulty per team over the next `horizon_days` days.

    Important details:
    - Ensure all datetimes used for comparisons are timezone-aware in UTC to avoid TypeError.
    - The FPL API provides 'team_h_difficulty' and 'team_a_difficulty' on fixtures.
    """
    # load into DataFrame and coerce kickoff times to UTC-aware datetimes
    fdf = pd.DataFrame(fixtures)
    if fdf.empty:
        return pd.DataFrame(columns=["team", "avg_difficulty", "fixtures_count"])  # empty schema

    # Parse kickoff_time as UTC-aware timestamps; errors -> NaT
    if "kickoff_time" in fdf.columns:
        fdf["kickoff"] = pd.to_datetime(fdf["kickoff_time"], utc=True, errors="coerce")
    else:
        fdf["kickoff"] = pd.NaT

    # define timezone-aware now and horizon
    now = pd.Timestamp.now(tz="UTC")
    horizon = now + pd.Timedelta(days=horizon_days)

    # filter upcoming fixtures that have a kickoff in the next horizon days
    upcoming = fdf[(fdf["kickoff"] >= now) & (fdf["kickoff"] <= horizon)].copy()

    # collect difficulty scores per team (using the fixture's provided difficulty)
    team_scores: Dict[int, List[int]] = {}
    for _, row in upcoming.iterrows():
        # team ids
        th = row.get("team_h")
        ta = row.get("team_a")
        # difficulties (some fixtures include 'team_h_difficulty' and 'team_a_difficulty')
        dh = row.get("team_h_difficulty", row.get("difficulty"))
        da = row.get("team_a_difficulty", row.get("difficulty"))
        try:
            if pd.notna(dh):
                team_scores.setdefault(int(th), []).append(int(dh))
            if pd.notna(da):
                team_scores.setdefault(int(ta), []).append(int(da))
        except Exception:
            # ignore malformed rows
            continue

    rows = []
    for team_id, scores in team_scores.items():
        avg = float(sum(scores)) / len(scores) if scores else None
        rows.append({"team": int(team_id), "avg_difficulty": avg, "fixtures_count": len(scores)})

    return pd.DataFrame(rows)

# --------------------
# Projected points model
# --------------------
def project_points(players_df: pd.DataFrame, fdr_df: pd.DataFrame) -> pd.DataFrame:
    """A simple, transparent projected points calculation.

    Formula (explainable heuristic):
      base = 0.6 * form + 0.4 * points_per_game
      difficulty_adj = 3.0 / avg_difficulty  # average difficulty baseline is 3.0
      proj = base * difficulty_adj

    We clip/handle missing values (default avg_difficulty=3.0 -> neutral), and scale for display.
    """
    df = players_df.copy()

    # create a lookup for avg_difficulty by team id
    fdr_lookup = dict(zip(fdr_df["team"], fdr_df["avg_difficulty"])) if not fdr_df.empty else {}

    def compute_row(r):
        base = 0.6 * float(r.get("form", 0.0)) + 0.4 * float(r.get("points_per_game", 0.0))
        avg_diff = fdr_lookup.get(int(r["team_id"]) if pd.notna(r["team_id"]) else -1, 3.0)
        # avoid division by zero or None
        if not avg_diff or pd.isna(avg_diff):
            avg_diff = 3.0
        difficulty_adj = 3.0 / float(avg_diff)
        proj = base * difficulty_adj
        # small boost for players who have higher minutes (more likely to play)
        minutes = float(r.get("minutes", 0.0))
        if minutes >= 90:
            proj *= 1.05
        return max(proj, 0.0)

    df["projected_points"] = df.apply(compute_row, axis=1)
    return df

# --------------------
# Squad validation and auto-fix
# --------------------

def validate_squad(squad_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate squad against FPL-like rules. Returns (is_valid, list_of_issues)."""
    issues: List[str] = []

    if squad_df.empty:
        issues.append("Squad is empty.")
        return False, issues

    # total value
    total_value = squad_df["price_m"].sum()
    if total_value > SQUAD_BUDGET + 0.001:
        issues.append(f"Budget exceeded: {total_value:.1f}m > {SQUAD_BUDGET}m")

    # per-team limit
    team_counts = squad_df["team_id"].value_counts().to_dict()
    for team_id, count in team_counts.items():
        if count > MAX_PER_TEAM:
            # resolve team id -> show value; team names may not be present here
            issues.append(f"Too many players from team id {team_id}: {count} (max {MAX_PER_TEAM})")

    # position counts
    pos_counts = squad_df["position"].value_counts().to_dict()
    for pos, required in POSITION_RULES.items():
        have = pos_counts.get(pos, 0)
        if have != required:
            issues.append(f"Position mismatch for {pos}: have {have}, require {required}")

    is_valid = len(issues) == 0
    return is_valid, issues


def suggest_single_transfer(squad_df: pd.DataFrame, players_df: pd.DataFrame, budget_left: float) -> Optional[Tuple[dict, dict, str]]:
    """Suggest a single transfer (out -> in) based on projected points and budget constraints.

    Heuristic used:
    - Rank squad players by projected_points per price (value) ascending (weakest value first)
    - Rank outside players by same metric descending (best value first)
    - Find the first pair where buying the outside player (price) fits the budget when selling the chosen player
      and the in-player is not violating team limits (simple check)

    Returns tuple (out_player_row, in_player_row, reason) or None if no good transfer found.
    """
    # ensure sets for quick checks
    squad_ids = set(squad_df["id"].tolist())

    # compute value metric (proj per million)
    squad_df = squad_df.copy()
    players_df = players_df.copy()

    squad_df["value"] = squad_df.apply(lambda r: r["projected_points"] / r["price_m"] if r["price_m"] > 0 else 0, axis=1)
    players_df["value"] = players_df.apply(lambda r: r["projected_points"] / r["price_m"] if r["price_m"] > 0 else 0, axis=1)

    # candidates to sell: weakest value (lowest projected per m)
    sell_candidates = squad_df.sort_values("value").to_dict("records")
    # candidates to buy: players not in squad, sorted by best value
    buy_candidates = players_df[~players_df["id"].isin(squad_ids)].sort_values("value", ascending=False).to_dict("records")

    # simple team counts to avoid >3 per team after transfer
    team_counts = squad_df["team_id"].value_counts().to_dict()

    for out in sell_candidates[:10]:
        out_price = float(out["price_m"]) if pd.notna(out["price_m"]) else 0.0
        # available funds after selling this player
        available_after_sale = budget_left + out_price
        for incand in buy_candidates[:200]:
            in_price = float(incand["price_m"]) if pd.notna(incand["price_m"]) else 0.0
            # must fit budget
            if in_price > available_after_sale + 0.001:
                continue
            # must not be same id or already in squad
            if incand["id"] in squad_ids:
                continue
            # team limit check: if new player's team would exceed 3
            new_team = int(incand["team_id"]) if pd.notna(incand["team_id"]) else None
            if new_team is not None:
                new_count = team_counts.get(new_team, 0) + 1
                # if the out player is from the same team, the net effect may be neutral
                if int(out["team_id"]) == new_team:
                    new_count -= 1
                if new_count > MAX_PER_TEAM:
                    continue
            # found suitable transfer; reason based on projected points gain
            gain = incand["projected_points"] - out["projected_points"]
            reason = f"Replace {out['web_name']} ({out['position']}) with {incand['web_name']} — projected gain {gain:.2f} pts"
            return out, incand, reason
    return None


def apply_auto_fix(squad_df: pd.DataFrame, players_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Attempt to auto-fix a squad until validation passes or no more fixes available.

    Strategy (greedy heuristics):
    - While budget exceeded: try to replace the worst value player with a cheaper better-value player
    - While team_count violations exist: replace extra players from offending team with best available from other teams
    - While position mismatches exist: try to swap from surplus positions to missing ones

    This is not guaranteed optimal but aims to quickly produce a valid squad.
    """
    squad = squad_df.copy()
    players_pool = players_df.copy()
    swaps_applied: List[str] = []

    # compute current budget and helper
    def current_budget_left(sq):
        return SQUAD_BUDGET - sq["price_m"].sum()

    # loop with a safety counter to avoid infinite loops
    attempts = 0
    MAX_ATTEMPTS = 12
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        valid, issues = validate_squad(squad)
        if valid:
            break  # squad fixed

        # try to fix budget first
        total_value = squad["price_m"].sum()
        if total_value > SQUAD_BUDGET + 0.001:
            # sell the worst value player and find a cheaper replacement
            transfer = suggest_single_transfer(squad, players_pool, budget_left=SQUAD_BUDGET - (squad["price_m"].sum() - 0))
            if transfer:
                out, incand, reason = transfer
                # apply swap
                swaps_applied.append(reason)
                squad = squad[squad["id"] != out["id"]].copy()
                squad = pd.concat([squad, pd.DataFrame([incand])], ignore_index=True, sort=False)
                continue
            else:
                # no transfer found to reduce budget; attempt replacing the most expensive player for a cheaper one
                most_expensive = squad.sort_values("price_m", ascending=False).iloc[0]
                candidates = players_pool[~players_pool["id"].isin(squad["id"]) & (players_pool["position"] == most_expensive["position"])]
                candidates = candidates[candidates["price_m"] <= most_expensive["price_m"] - 0.1]
                candidates = candidates.sort_values("projected_points", ascending=False)
                if not candidates.empty:
                    newp = candidates.iloc[0]
                    swaps_applied.append(f"Replace {most_expensive['web_name']} ({most_expensive['price_m']}m) with {newp['web_name']} ({newp['price_m']}m) to reduce budget")
                    squad = squad[squad["id"] != most_expensive["id"]].copy()
                    squad = pd.concat([squad, pd.DataFrame([newp])], ignore_index=True, sort=False)
                    continue
                else:
                    # cannot fix budget
                    break

        # fix team count violations
        team_counts = squad["team_id"].value_counts()
        offending = [t for t, c in team_counts.items() if c > MAX_PER_TEAM]
        if offending:
            for team_id in offending:
                # remove the worst value player from that team
                candidates = squad[squad["team_id"] == team_id]
                candidates = candidates.sort_values(by=["projected_points"], ascending=True)
                if candidates.empty:
                    continue
                out = candidates.iloc[0]
                # find replacement not in squad from different team and same position
                replacements = players_pool[~players_pool["id"].isin(squad["id"]) & (players_pool["position"] == out["position"]) & (players_pool["team_id"] != team_id)]
                replacements = replacements.sort_values(by=["projected_points"], ascending=False)
                if not replacements.empty:
                    newp = replacements.iloc[0]
                    swaps_applied.append(f"Replace {out['web_name']} from team {team_id} with {newp['web_name']} from team {int(newp['team_id'])}")
                    squad = squad[squad["id"] != out["id"]].copy()
                    squad = pd.concat([squad, pd.DataFrame([newp])], ignore_index=True, sort=False)
                    break
            continue

        # fix position mismatches (if too many of one position and too few of another)
        pos_counts = squad["position"].value_counts().to_dict()
        # find any missing positions
        missing = [pos for pos, req in POSITION_RULES.items() if pos_counts.get(pos, 0) < req]
        surplus = [pos for pos, cnt in pos_counts.items() if cnt > POSITION_RULES.get(pos, 0)]
        if missing and surplus:
            # try to replace one surplus player with best candidate in missing position
            out_pos = surplus[0]
            replace_candidates = squad[squad["position"] == out_pos].sort_values("projected_points").to_dict("records")
            if not replace_candidates:
                break
            out = replace_candidates[0]
            # find a candidate of missing position not in squad, affordable
            replacements = players_pool[~players_pool["id"].isin(squad["id"]) & (players_pool["position"] == missing[0])]
            replacements = replacements.sort_values("projected_points", ascending=False)
            if not replacements.empty:
                newp = replacements.iloc[0]
                swaps_applied.append(f"Swap {out['web_name']} ({out['position']}) for {newp['web_name']} ({newp['position']}) to fix positions")
                squad = squad[squad["id"] != out["id"]].copy()
                squad = pd.concat([squad, pd.DataFrame([newp])], ignore_index=True, sort=False)
                continue
        # if we get here, no more heuristic fixes available
        break

    # final validation
    valid, issues = validate_squad(squad)
    if not valid:
        swaps_applied.append("Auto-fix incomplete: some issues remain: " + "; ".join(issues))
    return squad.reset_index(drop=True), swaps_applied

# --------------------
# Persistence helpers
# --------------------

def save_squad_to_disk(squad: List[dict], path: str = SAVED_SQUAD_FILE) -> None:
    """Save squad (list of dicts) to a JSON file. Silently fails on IO errors but returns None."""
    try:
        with open(path, "w") as f:
            json.dump(squad, f)
    except Exception:
        # we intentionally don't raise to keep the app running; log could be added
        pass


def load_squad_from_disk(path: str = SAVED_SQUAD_FILE) -> List[dict]:
    """Load squad from disk; returns empty list on failure."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

# --------------------
# UI Pages
# --------------------

def page_player_explorer(players_df: pd.DataFrame, fixtures: List[dict]):
    st.header("Player Explorer")
    st.write("Filter and explore players. Table supports quick sorting and scanning.")

    # simple filters
    positions = ["All"] + sorted(players_df["position"].unique().tolist())
    teams = ["All"] + sorted(players_df["team_name"].unique().tolist())
    sel_pos = st.selectbox("Position", positions)
    sel_team = st.selectbox("Team", teams)
    name_search = st.text_input("Search name")

    df = players_df.copy()
    if sel_pos != "All":
        df = df[df["position"] == sel_pos]
    if sel_team != "All":
        df = df[df["team_name"] == sel_team]
    if name_search:
        df = df[df["web_name"].str.contains(name_search, case=False, na=False)]

    # let user choose columns to show
    display_cols = ["web_name", "team_name", "position", "price_m", "total_points", "form", "projected_points"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols].sort_values(by=["projected_points"], ascending=False).reset_index(drop=True))


def page_fixtures(fixtures: List[dict], horizon_days: int = 42):
    st.header("Fixtures & FDR")
    st.write("Upcoming fixtures and a simple Fixture Difficulty Rating (FDR) aggregated by team.")

    fdr_df = build_fdr(fixtures, horizon_days=horizon_days)
    if fdr_df.empty:
        st.info("No upcoming fixtures found in the selected horizon.")
        return

    # merge team names from bootstrap is done at caller; here we display team id + difficulty
    st.dataframe(fdr_df.sort_values(["avg_difficulty", "fixtures_count"]).reset_index(drop=True))


def page_projected_points(players_df: pd.DataFrame, fixtures: List[dict]):
    st.header("Projected Points")
    st.write("A simple projected points model based on form, points per game and fixture difficulty.")

    horizon = st.slider("Look ahead (days) for fixture difficulty", 7, 84, 21)
    fdr_df = build_fdr(fixtures, horizon_days=horizon)
    proj = project_points(players_df, fdr_df)

    display_cols = ["web_name", "team_name", "position", "price_m", "form", "points_per_game", "projected_points"]
    display_cols = [c for c in display_cols if c in proj.columns]

    st.dataframe(proj[display_cols].sort_values("projected_points", ascending=False).reset_index(drop=True).head(200))


def page_transfer_suggestions(players_df: pd.DataFrame, squad_df: pd.DataFrame):
    st.header("Transfer Suggestions")
    st.write("One-transfer suggestions based on projected points and simple heuristics.")

    budget_left = SQUAD_BUDGET - (squad_df["price_m"].sum() if not squad_df.empty else 0.0)
    st.write(f"Budget left (assuming current squad): {budget_left:.1f}m")

    suggestion = suggest_single_transfer(squad_df, players_df, budget_left)
    if suggestion:
        out, incand, reason = suggestion
        st.success(reason)
        st.write("Out:")
        st.write(out)
        st.write("In:")
        st.write(incand)
    else:
        st.info("No one-for-one upgrade found within budget using current heuristics.")


def page_my_team(players_df: pd.DataFrame, fixtures: List[dict]):
    st.header("My Team Importer & Validator")
    st.write("Paste player web_names (comma-separated) or upload a saved squad to import. Use Apply Auto-Fix to automatically fix rule violations.")

    # load any saved squad from disk into session_state once
    if "saved_loaded" not in st.session_state:
        st.session_state["saved_loaded"] = True
        saved = load_squad_from_disk()
        if saved:
            st.session_state.setdefault("my_squad", saved)

    # initialize session squad if missing
    if "my_squad" not in st.session_state:
        st.session_state["my_squad"] = []

    # Import area
    import_text = st.text_area("Paste player web_names (comma separated) OR leave empty to use saved squad", height=100)
    if st.button("Import from pasted names") and import_text.strip():
        names = [n.strip() for n in import_text.split(",") if n.strip()]
        # try to match by web_name (case-insensitive)
        matches = players_df[players_df["web_name"].str.lower().isin([n.lower() for n in names])]
        if matches.empty:
            st.warning("No matching players found for the pasted names. Ensure you used web_name (e.g., 'Salah').")
        else:
            # store in session as list of dicts
            st.session_state["my_squad"] = matches.to_dict("records")
            st.success(f"Imported {len(matches)} players into your squad (session-only until saved).")

    # Option to import by uploading JSON (saved_squad format)
    uploaded = st.file_uploader("Or upload a saved squad JSON file", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, list):
                st.session_state["my_squad"] = data
                st.success("Uploaded squad loaded into session.")
            else:
                st.error("Uploaded file format invalid. Expected a list of player dicts.")
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")

    # Display current squad
    squad_df = pd.DataFrame(st.session_state.get("my_squad", []))
    if squad_df.empty:
        st.info("No squad loaded. Use the import box above or save a new squad after building one.")
    else:
        st.subheader("Current Squad")
        # show key columns
        show_cols = [c for c in ["web_name", "team_name", "position", "price_m", "projected_points"] if c in squad_df.columns]
        st.dataframe(squad_df[show_cols].reset_index(drop=True))

        # run validation and show issues
        is_valid, issues = validate_squad(squad_df)
        if is_valid:
            st.success("Squad is valid against FPL rules.")
        else:
            st.warning("Squad issues detected:")
            for it in issues:
                st.write(f"- {it}")

            # auto-fix suggestion (preview)
            if st.button("Preview Auto-Fix Suggestions"):
                fixed, swaps = apply_auto_fix(squad_df, players_df)
                st.subheader("Auto-Fix Preview")
                if swaps:
                    for s in swaps:
                        st.write(f"- {s}")
                st.dataframe(fixed[[c for c in ["web_name", "team_name", "position", "price_m", "projected_points"] if c in fixed.columns]].reset_index(drop=True))

            # apply auto-fix
            if st.button("Apply Auto-Fix"):
                fixed, swaps = apply_auto_fix(squad_df, players_df)
                if swaps:
                    st.session_state["my_squad"] = fixed.to_dict("records")
                    st.success("Auto-Fix applied. Review your squad and Save to persist.")
                    for s in swaps:
                        st.write(f"- {s}")
                else:
                    st.info("No auto-fix swaps were possible.")

        # Save squad to disk and session
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save My Squad (session + disk)"):
                # ensure session contains serializable dicts
                save_squad_to_disk(st.session_state["my_squad"])  # persist on disk
                st.success("Squad saved to disk and session. It will be loaded next time the app starts.")
        with col2:
            if st.button("Clear Saved Squad"):
                st.session_state["my_squad"] = []
                try:
                    if os.path.exists(SAVED_SQUAD_FILE):
                        os.remove(SAVED_SQUAD_FILE)
                except Exception:
                    pass
                st.success("Cleared session squad and removed saved file (if it existed).")

# --------------------
# Main app wiring
# --------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # fetch data
    try:
        bootstrap = fetch_bootstrap()
        fixtures = fetch_fixtures()
    except Exception as e:
        st.error(f"Failed to fetch FPL data: {e}")
        st.stop()

    # build players DataFrame and compute projections
    players_df = build_players_df(bootstrap)
    # compute FDR for a medium horizon to feed projections (we also compute again on the projected page)
    fdr_df = build_fdr(fixtures, horizon_days=21)
    players_df = project_points(players_df, fdr_df)

    # load saved squad from disk into session if present
    if "my_squad" not in st.session_state:
        saved = load_squad_from_disk()
        st.session_state["my_squad"] = saved

    # navigation
    menu = ["Player Explorer", "Fixtures", "Projected Points", "Transfer Suggestions", "My Team"]
    choice = st.sidebar.radio("Navigate", menu)

    if choice == "Player Explorer":
        page_player_explorer(players_df, fixtures)
    elif choice == "Fixtures":
        st.header("Fixtures (raw)")
        # show basic fixture info; convert kickoff to readable local time string if present
        fdf = pd.DataFrame(fixtures)
        if "kickoff_time" in fdf.columns:
            fdf["kickoff"] = pd.to_datetime(fdf["kickoff_time"], utc=True, errors="coerce")
        st.dataframe(fdf[[c for c in ["event", "kickoff", "team_h", "team_a"] if c in fdf.columns]].reset_index(drop=True))
    elif choice == "Projected Points":
        page_projected_points(players_df, fixtures)
    elif choice == "Transfer Suggestions":
        # create a squad_df from session for suggestions
        squad_df = pd.DataFrame(st.session_state.get("my_squad", []))
        if squad_df.empty:
            st.info("Load or import a squad first (My Team page) to get transfer suggestions tailored to your team.")
        else:
            page_transfer_suggestions(players_df, squad_df)
    elif choice == "My Team":
        page_my_team(players_df, fixtures)

    st.markdown("---")
    st.write("Built as a helper tool — projections are heuristics, not guaranteed. Use for guidance and always check official FPL info before transferring.")


if __name__ == "__main__":
    main()
