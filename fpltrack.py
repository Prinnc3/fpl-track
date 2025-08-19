"""
Streamlit FPL App — Expanded
===========================

New Features:
- Navigation menu with multiple pages.
- Player Explorer & Fixtures (original MVP).
- Transfer Suggestion Engine.
- Projected Points (simple model).
- My Team Importer (manual input for now).
- UI polish for deployment.

How to run
----------
1. Install requirements:
   pip install streamlit requests pandas

2. Run app:
   streamlit run streamlit_fpl_app.py

Notes
-----
- Data from official FPL API.
- Transfer suggestion engine uses value (points per million) and fixtures.
- Projected points is a simple model based on form and fixture difficulty.
- My Team Importer allows manual input of your squad (paste player names).

"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List

st.set_page_config(page_title="FPL App", layout="wide")

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

    elements = elements.merge(teams, left_on='team', right_on='id', how='left', suffixes=('','_team'))
    elements = elements.merge(element_types, left_on='element_type', right_on='id', how='left', suffixes=('','_etype'))

    cols = ['id','first_name','second_name','web_name','team','name','short_name',
            'element_type','singular_name_short','now_cost','total_points','form','minutes','selected_by_percent']
    cols = [c for c in cols if c in elements.columns]

    df = elements[cols].copy()
    df['full_name'] = df['first_name'].fillna('') + ' ' + df['second_name'].fillna('')
    if 'now_cost' in df.columns:
        df['price_m'] = df['now_cost'] / 10
    df['form'] = pd.to_numeric(df['form'], errors='coerce')
    df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')
    df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce')
    df['ppM'] = df['total_points'] / df['price_m']
    df = df.rename(columns={'short_name':'team_name','singular_name_short':'position'})
    return df


def build_fdr(fixtures: List[dict], horizon_days: int = 42) -> pd.DataFrame:
    now = datetime.utcnow()
    horizon = now + timedelta(days=horizon_days)
    fdf = pd.DataFrame(fixtures).copy()
    if fdf.empty:
        return pd.DataFrame()
    if 'kickoff_time' in fdf.columns:
        fdf['kickoff'] = pd.to_datetime(fdf['kickoff_time'], utc=True, errors='coerce')
    else:
        fdf['kickoff'] = pd.NaT
    fdf_upcoming = fdf[(fdf['kickoff'] >= now) & (fdf['kickoff'] <= horizon)]

    team_scores = {}
    for _, row in fdf_upcoming.iterrows():
        try:
            th = int(row.get('team_h'))
            ta = int(row.get('team_a'))
        except Exception:
            continue
        dh = row.get('team_h_difficulty', row.get('difficulty'))
        da = row.get('team_a_difficulty', row.get('difficulty'))
        if pd.notna(dh):
            team_scores.setdefault(th, []).append(int(dh))
        if pd.notna(da):
            team_scores.setdefault(ta, []).append(int(da))

    rows = []
    for team, scores in team_scores.items():
        avg = sum(scores)/len(scores) if scores else None
        rows.append({'team': team, 'avg_difficulty': avg, 'fixtures_count': len(scores)})

    return pd.DataFrame(rows)


################################
# PAGE FUNCTIONS
################################

def page_player_explorer(players_df, fixtures, bootstrap):
    st.header("Player Explorer & Fixtures")

    col1, col2 = st.columns([1,3])
    with col1:
        pos_options = ['All'] + sorted(players_df['position'].dropna().unique().tolist())
        sel_pos = st.selectbox('Position', pos_options, index=0)
        team_options = ['All'] + sorted(players_df['team_name'].dropna().unique().tolist())
        sel_team = st.selectbox('Team', team_options, index=0)
        name_search = st.text_input('Search (name)')
        min_points = st.number_input('Min total points', value=0, step=1)
        sort_by = st.selectbox('Sort by', ['total_points','form','ppM','selected_by_percent','price_m'], index=0)
        top_n = st.slider('Show top N players', min_value=10, max_value=300, value=100)

    with col2:
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
        display_cols = ['web_name','team_name','position','price_m','total_points','form','ppM','selected_by_percent']
        st.dataframe(df[display_cols].head(top_n).reset_index(drop=True))

    st.subheader('Upcoming Fixtures & FDR')
    horizon_days = st.slider('Look ahead days', min_value=7, max_value=84, value=42)
    fdr_df = build_fdr(fixtures, horizon_days=horizon_days)
    teams_map = pd.DataFrame(bootstrap.get('teams', []))[['id','name','short_name']]
    if not fdr_df.empty:
        fdr_df = fdr_df.merge(teams_map, left_on='team', right_on='id', how='left')
        fdr_df = fdr_df[['short_name','name','avg_difficulty','fixtures_count']].rename(columns={'short_name':'team','name':'team_full'})
        st.dataframe(fdr_df.reset_index(drop=True))


def page_transfer_suggestions(players_df, fixtures):
    st.header("Transfer Suggestion Engine")
    budget = st.number_input("Available budget (in millions)", min_value=0.0, value=5.0, step=0.5)
    st.write("Top value players within budget:")
    df = players_df[players_df['price_m'] <= budget]
    df = df.sort_values('ppM', ascending=False)
    st.dataframe(df[['web_name','team_name','position','price_m','total_points','ppM']].head(10))


def page_projected_points(players_df, fixtures):
    st.header("Projected Points (Simple Model)")
    fdr_df = build_fdr(fixtures, horizon_days=21)
    df = players_df.merge(fdr_df, left_on='team', right_on='team', how='left')
    df['avg_difficulty'] = df['avg_difficulty'].fillna(3)
    df['proj_points'] = (df['form'].fillna(0).astype(float) * 2) / df['avg_difficulty']
    df = df.sort_values('proj_points', ascending=False)
    st.dataframe(df[['web_name','team_name','position','price_m','form','avg_difficulty','proj_points']].head(20))


def page_my_team(players_df):
    st.header("My Team Importer")
    team_input = st.text_area("Paste your team player names (comma separated)")
    if team_input:
        names = [n.strip() for n in team_input.split(',')]
        df = players_df[players_df['web_name'].isin(names)]
        if df.empty:
            st.warning("No matching players found.")
        else:
            st.subheader("Your Team Stats")
            st.dataframe(df[['web_name','team_name','position','price_m','total_points','form']])
            st.write(f"Total Team Points: {df['total_points'].sum()} | Total Value: {df['price_m'].sum()}m")


################################
# MAIN APP
################################

with st.spinner('Fetching FPL data...'):
    try:
        bootstrap = fetch_bootstrap()
        fixtures = fetch_fixtures()
        players_df = build_players_df(bootstrap)
    except Exception as e:
        st.error(f"Failed to fetch FPL data: {e}")
        st.stop()

menu = ["Player Explorer","Transfer Suggestions","Projected Points","My Team"]
choice = st.sidebar.radio("Navigate", menu)

if choice == "Player Explorer":
    page_player_explorer(players_df, fixtures, bootstrap)
elif choice == "Transfer Suggestions":
    page_transfer_suggestions(players_df, fixtures)
elif choice == "Projected Points":
    page_projected_points(players_df, fixtures)
elif choice == "My Team":
    page_my_team(players_df)

st.sidebar.markdown("---")
st.sidebar.write("FPL App MVP — enhanced with multiple features. Ready for deployment!")
