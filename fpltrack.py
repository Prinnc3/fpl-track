"""
Streamlit FPL MVP
=================

Features included:
- Fetches data from the official FPL API (bootstrap-static and fixtures).
- Player table with filters (position, team, name search) and sortable columns.
- Fixtures view showing upcoming fixtures and a simple Fixture Difficulty Rating (FDR) per team.
- Simple Captain suggestion based on player form and upcoming fixture difficulty.

How to run
----------
1. Install requirements:
   pip install streamlit requests pandas

2. Run app:
   streamlit run streamlit_fpl_mvp.py

Notes
-----
- The app fetches live data from the official FPL endpoints. If the FPL API changes structure, the fetching/parsing code may need small fixes.
- This is an MVP; we can add: transfer suggestions, value-for-money ranks, a "My Team" importer, projected points, and visual charts.

"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List

st.set_page_config(page_title="FPL MVP — Player Table & Fixtures", layout="wide")

API_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{API_BASE}/bootstrap-static/"
FIXTURES_URL = f"{API_BASE}/fixtures/"

@st.cache_data(ttl=60*10)
def fetch_bootstrap() -> dict:
    resp = requests.get(BOOTSTRAP_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=60*10)
def fetch_fixtures() -> List[dict]:
    resp = requests.get(FIXTURES_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_players_df(bootstrap: dict) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap.get('elements', []))
    teams = pd.DataFrame(bootstrap.get('teams', []))[['id','name','short_name']]
    element_types = pd.DataFrame(bootstrap.get('element_types', []))[['id','singular_name_short']]

    # Merge team and position names
    elements = elements.merge(teams, left_on='team', right_on='id', how='left', suffixes=('','_team'))
    elements = elements.merge(element_types, left_on='element_type', right_on='id', how='left', suffixes=('','_etype'))

    # Useful columns
    cols = ['id','first_name','second_name','web_name','team','name','short_name',
            'element_type','singular_name_short','now_cost','total_points','form','minutes','selected_by_percent','chance_of_playing_next_round']
    cols = [c for c in cols if c in elements.columns]

    df = elements[cols].copy()
    df['full_name'] = df['first_name'].fillna('') + ' ' + df['second_name'].fillna('')
    # cost in millions
    if 'now_cost' in df.columns:
        df['price_m'] = df['now_cost'] / 10
    # numeric conversions
    df['form'] = pd.to_numeric(df['form'], errors='coerce')
    df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')
    df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce')

    # value metric: points per million
    df['ppM'] = df['total_points'] / df['price_m']

    # rename for easier display
    df = df.rename(columns={'short_name':'team_name','singular_name_short':'position'})

    return df


def build_fdr(fixtures: List[dict], horizon_days: int = 42) -> pd.DataFrame:
    # horizon_days: look ahead N days for upcoming fixtures
    now = datetime.utcnow()
    horizon = now + timedelta(days=horizon_days)

    # DataFrame of fixtures with kickoff parsed
    fdf = pd.DataFrame(fixtures).copy()
    if fdf.empty:
        return pd.DataFrame()

    # parse kickoff_time if present
    if 'kickoff_time' in fdf.columns:
        fdf['kickoff'] = pd.to_datetime(fdf['kickoff_time'], utc=True, errors='coerce')
    else:
        fdf['kickoff'] = pd.NaT

    # Filter upcoming fixtures in horizon and not finished
    fdf_upcoming = fdf[(fdf['kickoff'] >= now) & (fdf['kickoff'] <= horizon)]

    # distance: use provided difficulty fields if present
    # Possible difficulty fieldnames: 'team_h_difficulty' and 'team_a_difficulty' (present in FPL fixtures)
    # We'll construct per-team list of difficulties
    team_scores = {}
    for _, row in fdf_upcoming.iterrows():
        # home team
        try:
            th = int(row.get('team_h'))
            ta = int(row.get('team_a'))
        except Exception:
            continue
        # difficulties (fallback to 'difficulty')
        dh = row.get('team_h_difficulty', row.get('difficulty'))
        da = row.get('team_a_difficulty', row.get('difficulty'))
        if pd.notna(dh):
            team_scores.setdefault(th, []).append(int(dh))
        if pd.notna(da):
            team_scores.setdefault(ta, []).append(int(da))

    # compute average difficulty per team
    rows = []
    for team, scores in team_scores.items():
        avg = sum(scores)/len(scores) if scores else None
        rows.append({'team': team, 'avg_difficulty': avg, 'fixtures_count': len(scores)})

    fdr_df = pd.DataFrame(rows)
    return fdr_df


#########################
# Streamlit UI
#########################

st.title("FPL MVP — Player Table & Fixtures")

with st.spinner('Fetching FPL data...'):
    try:
        bootstrap = fetch_bootstrap()
        fixtures = fetch_fixtures()
    except Exception as e:
        st.error(f"Failed to fetch FPL data: {e}")
        st.stop()

players_df = build_players_df(bootstrap)

# Layout: Left filters, right display
col1, col2 = st.columns([1,3])

with col1:
    st.header("Filters")
    pos_options = ['All'] + sorted(players_df['position'].dropna().unique().tolist())
    sel_pos = st.selectbox('Position', pos_options, index=0)

    team_options = ['All'] + sorted(players_df['team_name'].dropna().unique().tolist())
    sel_team = st.selectbox('Team', team_options, index=0)

    name_search = st.text_input('Search (name)')

    min_points = st.number_input('Min total points', value=0, step=1)
    sort_by = st.selectbox('Sort by', ['total_points','form','ppM','selected_by_percent','price_m'], index=0)
    top_n = st.slider('Show top N players', min_value=10, max_value=300, value=100)

    st.markdown("---")
    st.write("Captain suggestion (from a selected pool):")
    selected_pool = st.multiselect('Select players to consider for captain', players_df['web_name'].tolist(), max_selections=10)
    if st.button('Suggest Captain'):
        if not selected_pool:
            st.warning('Pick at least one player to consider')
        else:
            pool_df = players_df[players_df['web_name'].isin(selected_pool)].copy()
            # Simple score: form / avg_difficulty (we'll map team difficulty to 1..5 scale if available)
            # Get FDR for teams (short horizon)
            fdr_df = build_fdr(fixtures, horizon_days=21)
            pool_df = pool_df.merge(fdr_df, left_on='team', right_on='team', how='left')
            pool_df['avg_difficulty'] = pool_df['avg_difficulty'].fillna(3)
            pool_df['captain_score'] = (pool_df['form'].fillna(0).astype(float) + pool_df['total_points'].fillna(0).astype(float)/20) / pool_df['avg_difficulty']
            best = pool_df.sort_values('captain_score', ascending=False).iloc[0]
            st.success(f"Suggested captain: {best['web_name']} ({best['team_name']}) — score {best['captain_score']:.2f}")

with col2:
    st.header('Player Explorer')

    df = players_df.copy()
    if sel_pos != 'All':
        df = df[df['position'] == sel_pos]
    if sel_team != 'All':
        df = df[df['team_name'] == sel_team]
    if name_search:
        df = df[df['web_name'].str.contains(name_search, case=False, na=False) | df['full_name'].str.contains(name_search, case=False, na=False)]

    df = df[df['total_points'] >= min_points]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    display_cols = ['web_name','full_name','team_name','position','price_m','total_points','form','ppM','selected_by_percent']
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols].head(top_n).reset_index(drop=True))

    st.markdown('---')
    st.write('Quick top-10 by selected metric:')
    metric = st.selectbox('Metric for top list', ['total_points','form','ppM','selected_by_percent'], index=0)
    if metric in df.columns:
        st.table(df.sort_values(metric, ascending=False)[display_cols].head(10).reset_index(drop=True))


st.header('Upcoming Fixtures & FDR')

horizon_days = st.slider('Look ahead days for fixtures', min_value=7, max_value=84, value=42)

fdr_df = build_fdr(fixtures, horizon_days=horizon_days)

# Merge team names
teams_map = pd.DataFrame(bootstrap.get('teams', []))[['id','name','short_name']]
if not fdr_df.empty:
    fdr_df = fdr_df.merge(teams_map, left_on='team', right_on='id', how='left')
    fdr_df = fdr_df[['short_name','name','avg_difficulty','fixtures_count']].rename(columns={'short_name':'team','name':'team_full'})
    fdr_df = fdr_df.sort_values(['avg_difficulty','fixtures_count'])
    # Display with simple color hint via markdown
    def difficulty_label(x):
        if pd.isna(x):
            return 'N/A'
        x = float(x)
        if x <= 2:
            return 'Easy (<=2)'
        elif x <= 3:
            return 'Medium (2-3)'
        else:
            return 'Hard (>3)'

    fdr_df['difficulty_label'] = fdr_df['avg_difficulty'].apply(difficulty_label)
    st.dataframe(fdr_df.reset_index(drop=True))
else:
    st.info('No upcoming fixtures found in the chosen horizon.')

st.markdown('---')
st.write('Notes & next steps:')
st.write('- Add "My Team" import (copy-paste or FPL login) to run optimizers against your actual squad.')
st.write('- Add projected points model and transfer suggestion engine.')
st.write('- Add visual charts for form and ownership trends.')

st.write('Built as an MVP — tell me which feature you want next (transfer engine, projected points, or team optimizer).')
