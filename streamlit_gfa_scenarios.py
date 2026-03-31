import itertools
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Utils
# =========================
def canonical_team(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()

    repl = {
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "À": "A", "Â": "A", "Ä": "A",
        "Ù": "U", "Û": "U", "Ü": "U",
        "Ô": "O", "Ö": "O",
        "Î": "I", "Ï": "I",
        "Ç": "C", "’": "'", "–": "-", "—": "-",
    }
    for a, b in repl.items():
        s = s.replace(a, b)

    keep = []
    for ch in s:
        if ch.isalnum() or ch in " -'/":
            keep.append(ch)
        else:
            keep.append(" ")
    s = "".join(keep)

    for token in ["AS ", "US ", "FC ", "AC ", "SC ", "RC ", "OLYMPIQUE ", "STADE "]:
        s = s.replace(token, "")

    s = s.replace("SAINT ", "ST ").replace("SAINTE ", "STE ")
    s = s.replace("RUMILLY-VALLIERES", "RUMILLY VALLIERES")
    s = s.replace("GFA RUMILLY VALLIERES", "RUMILLY VALLIERES")
    s = s.replace("LUSITANOS SAINT MAUR", "LUSITANOS ST MAUR")
    s = s.replace("ST-MAUR", "ST MAUR")
    s = s.replace("NIMES OLYMPIQUE", "NIMES")
    s = s.replace("FREJUS SAINT RAPHAEL", "FREJUS ST RAPHAEL")
    s = s.replace("FREJUS ST-RAPHAEL", "FREJUS ST RAPHAEL")
    s = s.replace("CRETEIL LUSITANOS", "CRETEIL")
    s = s.replace("GOAL", "GOAL FC")
    s = s.replace("ROUSSET", "ROUSSET")
    s = s.replace("BOBIGNY", "BOBIGNY")

    s = " ".join(s.split())
    return s


def validate_fixtures_vs_standings(standings_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> None:
    teams_standings = set(standings_df["team_key"])
    teams_fixtures = set(fixtures_df["home_key"]) | set(fixtures_df["away_key"])
    missing = sorted(teams_fixtures - teams_standings)
    if missing:
        raise ValueError(
            "Certaines équipes du calendrier ne sont pas dans le classement : "
            + ", ".join(missing)
        )


def apply_match_result(points: Dict[str, int], home_key: str, away_key: str, outcome: str) -> None:
    if outcome == "H":
        points[home_key] += 3
    elif outcome == "D":
        points[home_key] += 1
        points[away_key] += 1
    elif outcome == "A":
        points[away_key] += 3
    else:
        raise ValueError(f"Outcome inconnu: {outcome}")


# =========================
# Default demo data
# =========================
def make_default_standings() -> pd.DataFrame:
    data = [
        ("Nimes", 44),
        ("Cannes", 42),
        ("Lusitanos St Maur", 41),
        ("Rumilly Vallieres", 37),
        ("Hyeres", 36),
        ("Istres", 34),
        ("Creteil", 33),
        ("Andrezieux", 32),
        ("Goal FC", 28),
        ("Frejus St Raphael", 28),
        ("Bobigny", 26),
        ("Grasse", 26),
        ("Saint Priest", 24),
        ("Limonest", 22),
        ("Toulon", 22),
        ("FC Rousset SVO", 11),
    ]
    df = pd.DataFrame(data, columns=["team", "points"])
    df["played"] = 23
    df["gf"] = 0
    df["ga"] = 0
    return df


def make_default_fixtures() -> pd.DataFrame:
    rows = [
        (24, "Andrezieux", "Toulon"),
        (24, "Cannes", "Bobigny"),
        (24, "Istres", "Frejus St Raphael"),
        (24, "Limonest", "Rumilly Vallieres"),
        (24, "FC Rousset SVO", "Nimes"),
        (24, "Hyeres", "Grasse"),
        (24, "Lusitanos St Maur", "Creteil"),
        (24, "Saint Priest", "Goal FC"),

        (25, "Creteil", "Hyeres"),
        (25, "Frejus St Raphael", "Limonest"),
        (25, "Bobigny", "Andrezieux"),
        (25, "Rumilly Vallieres", "Saint Priest"),
        (25, "Goal FC", "FC Rousset SVO"),
        (25, "Nimes", "Lusitanos St Maur"),
        (25, "Grasse", "Cannes"),
        (25, "Toulon", "Istres"),

        (26, "Cannes", "Creteil"),
        (26, "Istres", "Bobigny"),
        (26, "Andrezieux", "Grasse"),
        (26, "Limonest", "Toulon"),
        (26, "Rumilly Vallieres", "Frejus St Raphael"),
        (26, "Hyeres", "Nimes"),
        (26, "Lusitanos St Maur", "Goal FC"),
        (26, "Saint Priest", "FC Rousset SVO"),

        (27, "Goal FC", "Hyeres"),
        (27, "Creteil", "Andrezieux"),
        (27, "Frejus St Raphael", "Saint Priest"),
        (27, "Bobigny", "Limonest"),
        (27, "FC Rousset SVO", "Lusitanos St Maur"),
        (27, "Nimes", "Cannes"),
        (27, "Grasse", "Istres"),
        (27, "Toulon", "Rumilly Vallieres"),

        (28, "Istres", "Creteil"),
        (28, "Cannes", "Goal FC"),
        (28, "Andrezieux", "Nimes"),
        (28, "Frejus St Raphael", "Toulon"),
        (28, "Limonest", "Grasse"),
        (28, "Rumilly Vallieres", "Bobigny"),
        (28, "Hyeres", "FC Rousset SVO"),
        (28, "Saint Priest", "Lusitanos St Maur"),

        (29, "Creteil", "Limonest"),
        (29, "Bobigny", "Frejus St Raphael"),
        (29, "FC Rousset SVO", "Cannes"),
        (29, "Goal FC", "Andrezieux"),
        (29, "Nimes", "Istres"),
        (29, "Grasse", "Rumilly Vallieres"),
        (29, "Lusitanos St Maur", "Hyeres"),
        (29, "Saint Priest", "Toulon"),

        (30, "Cannes", "Lusitanos St Maur"),
        (30, "Andrezieux", "FC Rousset SVO"),
        (30, "Frejus St Raphael", "Grasse"),
        (30, "Limonest", "Nimes"),
        (30, "Rumilly Vallieres", "Creteil"),
        (30, "Hyeres", "Saint Priest"),
        (30, "Istres", "Goal FC"),
        (30, "Toulon", "Bobigny"),
    ]
    return pd.DataFrame(rows, columns=["matchday", "home_team", "away_team"])


def make_default_h2h() -> pd.DataFrame:
    rows = [
        ("Rumilly Vallieres", "Cannes", 4, 1, 3, -3),
        ("Rumilly Vallieres", "Lusitanos St Maur", 4, 1, 1, -1),
        ("Rumilly Vallieres", "Nimes", 6, 0, 3, -3),
    ]
    return pd.DataFrame(
        rows,
        columns=["team_a", "team_b", "h2h_pts_a", "h2h_pts_b", "h2h_gd_a", "h2h_gd_b"],
    )


# =========================
# Standardization
# =========================
def standardize_standings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        lc = c.lower().strip()
        if lc in ("equipe", "club", "team_name"):
            rename[c] = "team"
        elif lc in ("pts", "point", "points_actuels"):
            rename[c] = "points"
        elif lc in ("j", "mj", "played"):
            rename[c] = "played"
        elif lc in ("bp", "gf"):
            rename[c] = "gf"
        elif lc in ("bc", "ga"):
            rename[c] = "ga"

    out = out.rename(columns=rename)

    if "team" not in out.columns or "points" not in out.columns:
        raise ValueError("Le classement doit contenir au moins 'team' et 'points'.")

    if "played" not in out.columns:
        out["played"] = np.nan
    if "gf" not in out.columns:
        out["gf"] = 0
    if "ga" not in out.columns:
        out["ga"] = 0

    out["team_key"] = out["team"].map(canonical_team)
    out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0).astype(int)
    out["played"] = pd.to_numeric(out["played"], errors="coerce")
    out["gf"] = pd.to_numeric(out["gf"], errors="coerce").fillna(0).astype(int)
    out["ga"] = pd.to_numeric(out["ga"], errors="coerce").fillna(0).astype(int)

    out = out.drop_duplicates("team_key").reset_index(drop=True)
    return out[["team", "team_key", "points", "played", "gf", "ga"]]


def standardize_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        lc = c.lower().strip()
        if lc in ("journee", "j", "matchday"):
            rename[c] = "matchday"
        elif lc in ("home", "dom", "domicile", "home_team"):
            rename[c] = "home_team"
        elif lc in ("away", "ext", "exterieur", "extérieur", "away_team"):
            rename[c] = "away_team"

    out = out.rename(columns=rename)

    if "matchday" not in out.columns or "home_team" not in out.columns or "away_team" not in out.columns:
        raise ValueError("Le calendrier doit contenir 'matchday', 'home_team', 'away_team'.")

    out["home_key"] = out["home_team"].map(canonical_team)
    out["away_key"] = out["away_team"].map(canonical_team)
    out["match_id"] = np.arange(len(out))
    return out[["match_id", "matchday", "home_team", "away_team", "home_key", "away_key"]]


def standardize_h2h(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "team_a", "team_b", "team_a_key", "team_b_key",
            "h2h_pts_a", "h2h_pts_b", "h2h_gd_a", "h2h_gd_b"
        ])

    out = df.copy()
    rename = {}
    for c in out.columns:
        lc = c.lower().strip()
        if lc in ("team_a", "equipe_a"):
            rename[c] = "team_a"
        elif lc in ("team_b", "equipe_b"):
            rename[c] = "team_b"
        elif lc in ("pts_a", "h2h_pts_a"):
            rename[c] = "h2h_pts_a"
        elif lc in ("pts_b", "h2h_pts_b"):
            rename[c] = "h2h_pts_b"
        elif lc in ("gd_a", "h2h_gd_a"):
            rename[c] = "h2h_gd_a"
        elif lc in ("gd_b", "h2h_gd_b"):
            rename[c] = "h2h_gd_b"

    out = out.rename(columns=rename)

    for c in ["h2h_pts_a", "h2h_pts_b", "h2h_gd_a", "h2h_gd_b"]:
        if c not in out.columns:
            out[c] = 0

    if "team_a" not in out.columns or "team_b" not in out.columns:
        return pd.DataFrame(columns=[
            "team_a", "team_b", "team_a_key", "team_b_key",
            "h2h_pts_a", "h2h_pts_b", "h2h_gd_a", "h2h_gd_b"
        ])

    out["team_a_key"] = out["team_a"].map(canonical_team)
    out["team_b_key"] = out["team_b"].map(canonical_team)

    for c in ["h2h_pts_a", "h2h_pts_b", "h2h_gd_a", "h2h_gd_b"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    return out[[
        "team_a", "team_b", "team_a_key", "team_b_key",
        "h2h_pts_a", "h2h_pts_b", "h2h_gd_a", "h2h_gd_b"
    ]]


# =========================
# Ranking
# =========================
def h2h_lookup(h2h_df: pd.DataFrame, a_key: str, b_key: str) -> Tuple[int, int, int, int]:
    if h2h_df.empty:
        return 0, 0, 0, 0

    row = h2h_df[(h2h_df["team_a_key"] == a_key) & (h2h_df["team_b_key"] == b_key)]
    if len(row) == 1:
        r = row.iloc[0]
        return int(r["h2h_pts_a"]), int(r["h2h_pts_b"]), int(r["h2h_gd_a"]), int(r["h2h_gd_b"])

    row = h2h_df[(h2h_df["team_a_key"] == b_key) & (h2h_df["team_b_key"] == a_key)]
    if len(row) == 1:
        r = row.iloc[0]
        return int(r["h2h_pts_b"]), int(r["h2h_pts_a"]), int(r["h2h_gd_b"]), int(r["h2h_gd_a"])

    return 0, 0, 0, 0


def rank_table(points: Dict[str, int], name_map: Dict[str, str], h2h_df: pd.DataFrame) -> pd.DataFrame:
    ordered = sorted(points.keys(), key=lambda k: (-points[k], canonical_team(name_map[k])))

    changed = True
    while changed:
        changed = False
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            if points[a] != points[b]:
                continue

            a_pts, b_pts, a_gd, b_gd = h2h_lookup(h2h_df, a, b)
            if b_pts > a_pts or (b_pts == a_pts and b_gd > a_gd):
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
                changed = True

    return pd.DataFrame({
        "rank": np.arange(1, len(ordered) + 1),
        "team_key": ordered,
        "team": [name_map[k] for k in ordered],
        "points": [points[k] for k in ordered],
    })


# =========================
# Exact mode
# =========================
def simulate_points_for_path(
    standings_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
    target_team: str,
    path_results: Optional[List[str]] = None,
    relevant_only: bool = True,
    relevant_teams: Optional[List[str]] = None,
    exhaustive_limit: int = 14,
    max_examples: int = 300,
) -> Dict[str, object]:
    target_key = canonical_team(target_team)
    name_map = dict(zip(standings_df["team_key"], standings_df["team"]))
    base_points = dict(zip(standings_df["team_key"], standings_df["points"]))

    fx = fixtures_df.copy()
    if relevant_only and relevant_teams:
        rel_keys = {canonical_team(x) for x in relevant_teams}
        rel_keys.add(target_key)
        fx = fx[(fx["home_key"].isin(rel_keys)) | (fx["away_key"].isin(rel_keys))].copy()

    target_matches = fx[
        (fx["home_key"] == target_key) | (fx["away_key"] == target_key)
    ].sort_values(["matchday", "match_id"]).reset_index(drop=True)

    other_matches = fx[~fx["match_id"].isin(target_matches["match_id"])].copy()

    path_results = path_results or [None] * len(target_matches)
    if len(path_results) != len(target_matches):
        raise ValueError("Nombre de résultats imposés incohérent.")

    fixed_points = base_points.copy()
    target_detail_rows = []
    free_target_matches_idx = []

    for idx, row in target_matches.iterrows():
        res = path_results[idx]
        home, away = row["home_key"], row["away_key"]

        if res is None:
            free_target_matches_idx.append(idx)
            continue

        if target_key == home:
            if res == "V":
                fixed_points[home] += 3
                lab = "Victoire"
            elif res == "N":
                fixed_points[home] += 1
                fixed_points[away] += 1
                lab = "Nul"
            else:
                fixed_points[away] += 3
                lab = "Défaite"
        else:
            if res == "V":
                fixed_points[away] += 3
                lab = "Victoire"
            elif res == "N":
                fixed_points[home] += 1
                fixed_points[away] += 1
                lab = "Nul"
            else:
                fixed_points[home] += 3
                lab = "Défaite"

        target_detail_rows.append({
            "matchday": row["matchday"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "resultat_impose": lab,
        })

    free_target_matches = target_matches.iloc[free_target_matches_idx].copy()
    matches_to_enum = pd.concat([free_target_matches, other_matches], ignore_index=True)

    n_matches = len(matches_to_enum)
    total_scenarios = 3 ** n_matches

    if n_matches > exhaustive_limit:
        return {
            "exact": False,
            "n_matches": n_matches,
            "total_scenarios": total_scenarios,
            "message": (
                f"Trop de matchs libres : {n_matches} "
                f"(soit {total_scenarios:,} scénarios). "
                "Réduis le périmètre ou impose un parcours GFA."
            ),
            "target_matches": pd.DataFrame(target_detail_rows),
        }

    favorable_solo = 0
    favorable_tied = 0
    examples = []
    target_distribution = {}

    for scenario in itertools.product(("H", "D", "A"), repeat=n_matches):
        pts = fixed_points.copy()
        scenario_rows = []

        for row, outc in zip(matches_to_enum.itertuples(index=False), scenario):
            apply_match_result(pts, row.home_key, row.away_key, outc)
            scenario_rows.append({
                "matchday": row.matchday,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "outcome": outc,
            })

        ranked = rank_table(pts, name_map, h2h_df)
        tgt_pts = pts[target_key]
        target_distribution[tgt_pts] = target_distribution.get(tgt_pts, 0) + 1

        top_points = ranked.iloc[0]["points"]
        leaders = ranked[ranked["points"] == top_points]["team_key"].tolist()

        if target_key in leaders:
            favorable_tied += 1
            if len(leaders) == 1:
                favorable_solo += 1

            if len(examples) < max_examples:
                examples.append({
                    "matches": pd.DataFrame(scenario_rows),
                    "ranking": ranked.copy(),
                    "leaders_count": len(leaders),
                })

    dist_df = pd.DataFrame(
        [{"points_finaux_cible": k, "nb_scenarios": v} for k, v in sorted(target_distribution.items())]
    )

    return {
        "exact": True,
        "n_matches": n_matches,
        "total_scenarios": total_scenarios,
        "favorable_solo": favorable_solo,
        "favorable_tied": favorable_tied,
        "probability_solo": favorable_solo / total_scenarios if total_scenarios else 0.0,
        "probability_tied": favorable_tied / total_scenarios if total_scenarios else 0.0,
        "target_matches": pd.DataFrame(target_detail_rows),
        "target_distribution": dist_df,
        "examples": examples,
    }


# =========================
# Monte Carlo mode
# =========================
def simulate_monte_carlo(
    standings_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
    target_team: str,
    path_results: Optional[List[str]] = None,
    relevant_only: bool = True,
    relevant_teams: Optional[List[str]] = None,
    n_simulations: int = 50000,
    max_examples: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    random.seed(seed)

    target_key = canonical_team(target_team)
    name_map = dict(zip(standings_df["team_key"], standings_df["team"]))
    base_points = dict(zip(standings_df["team_key"], standings_df["points"]))

    fx = fixtures_df.copy()
    if relevant_only and relevant_teams:
        rel_keys = {canonical_team(x) for x in relevant_teams}
        rel_keys.add(target_key)
        fx = fx[(fx["home_key"].isin(rel_keys)) | (fx["away_key"].isin(rel_keys))].copy()

    target_matches = fx[
        (fx["home_key"] == target_key) | (fx["away_key"] == target_key)
    ].sort_values(["matchday", "match_id"]).reset_index(drop=True)

    other_matches = fx[~fx["match_id"].isin(target_matches["match_id"])].copy()

    path_results = path_results or [None] * len(target_matches)
    if len(path_results) != len(target_matches):
        raise ValueError("Nombre de résultats imposés incohérent.")

    fixed_points = base_points.copy()
    target_detail_rows = []
    free_target_matches_idx = []

    for idx, row in target_matches.iterrows():
        res = path_results[idx]
        home, away = row["home_key"], row["away_key"]

        if res is None:
            free_target_matches_idx.append(idx)
            continue

        if target_key == home:
            if res == "V":
                fixed_points[home] += 3
                lab = "Victoire"
            elif res == "N":
                fixed_points[home] += 1
                fixed_points[away] += 1
                lab = "Nul"
            else:
                fixed_points[away] += 3
                lab = "Défaite"
        else:
            if res == "V":
                fixed_points[away] += 3
                lab = "Victoire"
            elif res == "N":
                fixed_points[home] += 1
                fixed_points[away] += 1
                lab = "Nul"
            else:
                fixed_points[home] += 3
                lab = "Défaite"

        target_detail_rows.append({
            "matchday": row["matchday"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "resultat_impose": lab,
        })

    free_target_matches = target_matches.iloc[free_target_matches_idx].copy()
    matches_to_sim = pd.concat([free_target_matches, other_matches], ignore_index=True)

    favorable_solo = 0
    favorable_tied = 0
    examples = []
    target_distribution = {}

    for _ in range(n_simulations):
        pts = fixed_points.copy()
        scenario_rows = []

        for row in matches_to_sim.itertuples(index=False):
            outc = random.choice(("H", "D", "A"))
            apply_match_result(pts, row.home_key, row.away_key, outc)

            scenario_rows.append({
                "matchday": row.matchday,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "outcome": outc,
            })

        ranked = rank_table(pts, name_map, h2h_df)
        tgt_pts = pts[target_key]
        target_distribution[tgt_pts] = target_distribution.get(tgt_pts, 0) + 1

        top_points = ranked.iloc[0]["points"]
        leaders = ranked[ranked["points"] == top_points]["team_key"].tolist()

        if target_key in leaders:
            favorable_tied += 1
            if len(leaders) == 1:
                favorable_solo += 1

            if len(examples) < max_examples:
                examples.append({
                    "matches": pd.DataFrame(scenario_rows),
                    "ranking": ranked.copy(),
                    "leaders_count": len(leaders),
                })

    dist_df = pd.DataFrame(
        [{"points_finaux_cible": k, "nb_simulations": v} for k, v in sorted(target_distribution.items())]
    )

    return {
        "exact": False,
        "n_simulations": n_simulations,
        "favorable_solo": favorable_solo,
        "favorable_tied": favorable_tied,
        "probability_solo": favorable_solo / n_simulations if n_simulations else 0.0,
        "probability_tied": favorable_tied / n_simulations if n_simulations else 0.0,
        "target_matches": pd.DataFrame(target_detail_rows),
        "target_distribution": dist_df,
        "examples": examples,
        "n_matches_simulated": len(matches_to_sim),
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Scénarios GFA 1er", layout="wide")
st.title("App Streamlit - scénarios où le GFA finit 1er")
st.caption("Mode exact si possible, sinon Monte Carlo pour obtenir le % de 1ère place.")

with st.sidebar:
    mode = st.radio("Source", ["Démo intégrée", "Uploader mes CSV"], index=0)

    if mode == "Uploader mes CSV":
        standings_file = st.file_uploader("Classement", type=["csv"])
        fixtures_file = st.file_uploader("Calendrier restant", type=["csv"])
        h2h_file = st.file_uploader("H2H (optionnel)", type=["csv"])

        standings_df = standardize_standings(
            pd.read_csv(standings_file) if standings_file is not None else make_default_standings()
        )
        fixtures_df = standardize_fixtures(
            pd.read_csv(fixtures_file) if fixtures_file is not None else make_default_fixtures()
        )
        h2h_df = standardize_h2h(
            pd.read_csv(h2h_file) if h2h_file is not None else make_default_h2h()
        )
    else:
        standings_df = standardize_standings(make_default_standings())
        fixtures_df = standardize_fixtures(make_default_fixtures())
        h2h_df = standardize_h2h(make_default_h2h())

    validate_fixtures_vs_standings(standings_df, fixtures_df)

    teams = standings_df.sort_values(["points", "team"], ascending=[False, True])["team"].tolist()
    default_target = "Rumilly Vallieres" if "Rumilly Vallieres" in teams else teams[0]
    target_team = st.selectbox("Équipe cible", teams, index=teams.index(default_target))

    relevant_only = st.checkbox("Limiter aux équipes suivies", value=True)
    default_rivals = [x for x in ["Cannes", "Lusitanos St Maur", "Nimes"] if x in teams]
    relevant_teams = st.multiselect("Équipes suivies", teams, default=default_rivals)

    calc_mode = st.radio("Mode de calcul", ["Monte Carlo", "Exact"], index=0)

    n_simulations = st.slider("Nombre de simulations", 1000, 200000, 50000, step=1000)
    seed = st.number_input("Seed Monte Carlo", min_value=0, value=42, step=1)

    exhaustive_limit = st.slider("Limite d'énumération exacte", 8, 18, 14)
    max_examples = st.slider("Exemples favorables stockés", 20, 1000, 200)

st.subheader("Classement actuel")
st.dataframe(standings_df[["team", "points", "played", "gf", "ga"]], use_container_width=True)

st.subheader("Calendrier restant")
st.dataframe(fixtures_df[["matchday", "home_team", "away_team"]], use_container_width=True)

target_key = canonical_team(target_team)
target_matches = fixtures_df[
    (fixtures_df["home_key"] == target_key) | (fixtures_df["away_key"] == target_key)
].sort_values(["matchday", "match_id"]).reset_index(drop=True)

st.subheader(f"Parcours imposé pour {target_team}")
st.write("Laisse 'Libre' pour ne pas imposer le match, ou fixe V / N / D pour réduire l'arbre.")
path_results = []

cols = st.columns(max(1, min(4, len(target_matches) if len(target_matches) else 1)))
for i, row in target_matches.iterrows():
    with cols[i % len(cols)]:
        val = st.selectbox(
            f"J{int(row['matchday'])} - {row['home_team']} vs {row['away_team']}",
            ["Libre", "V", "N", "D"],
            index=0,
            key=f"path_{i}",
        )
        path_results.append(None if val == "Libre" else val)

run = st.button("Lancer le calcul", type="primary")

if run:
    if calc_mode == "Exact":
        result = simulate_points_for_path(
            standings_df=standings_df,
            fixtures_df=fixtures_df,
            h2h_df=h2h_df,
            target_team=target_team,
            path_results=path_results,
            relevant_only=relevant_only,
            relevant_teams=relevant_teams,
            exhaustive_limit=exhaustive_limit,
            max_examples=max_examples,
        )
    else:
        result = simulate_monte_carlo(
            standings_df=standings_df,
            fixtures_df=fixtures_df,
            h2h_df=h2h_df,
            target_team=target_team,
            path_results=path_results,
            relevant_only=relevant_only,
            relevant_teams=relevant_teams,
            n_simulations=n_simulations,
            max_examples=max_examples,
            seed=int(seed),
        )

    st.subheader("Résultats")

    if calc_mode == "Exact" and not result["exact"]:
        st.error(result["message"])
    else:
        c1, c2, c3, c4 = st.columns(4)

        if calc_mode == "Exact":
            c1.metric("Scénarios totaux", f"{result['total_scenarios']:,}".replace(",", " "))
        else:
            c1.metric("Simulations", f"{result['n_simulations']:,}".replace(",", " "))

        c2.metric("1er seul", f"{100 * result['probability_solo']:.2f} %")
        c3.metric("1er ou ex aequo", f"{100 * result['probability_tied']:.2f} %")
        c4.metric("Scénarios favorables (co-1er inclus)", f"{result['favorable_tied']:,}".replace(",", " "))

        st.success(
            f"{target_team} finit 1er seul dans {100 * result['probability_solo']:.2f} % des cas "
            f"et 1er ou co-1er dans {100 * result['probability_tied']:.2f} % des cas."
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Résumé staff",
            "Distribution points",
            "Exemples favorables",
            "Matchs imposés",
        ])

        with tab1:
            st.dataframe(
                standings_df.sort_values("points", ascending=False).head(6)[["team", "points"]],
                use_container_width=True
            )
            if calc_mode == "Monte Carlo":
                st.write(f"Nombre de matchs simulés : {result['n_matches_simulated']}")

        with tab2:
            st.dataframe(result["target_distribution"], use_container_width=True)

        with tab3:
            if result["examples"]:
                choice = st.number_input(
                    "Exemple",
                    min_value=1,
                    max_value=len(result["examples"]),
                    value=1,
                    step=1
                )
                ex = result["examples"][choice - 1]
                st.write("Résultats")
                st.dataframe(ex["matches"], use_container_width=True)
                st.write("Classement final")
                st.dataframe(ex["ranking"][["rank", "team", "points"]], use_container_width=True)
            else:
                st.info("Aucun exemple favorable stocké.")

        with tab4:
            if not result["target_matches"].empty:
                st.dataframe(result["target_matches"], use_container_width=True)
            else:
                st.info("Aucun match imposé.")

st.markdown(
    '''
---
**Colonnes CSV attendues**
- Classement : `team, points, played, gf, ga`
- Calendrier : `matchday, home_team, away_team`
- H2H : `team_a, team_b, h2h_pts_a, h2h_pts_b, h2h_gd_a, h2h_gd_b`

**Notes**
- Le mode Exact est possible seulement si peu de matchs libres.
- Le mode Monte Carlo est recommandé pour le vrai calendrier.
- Les égalités à 2 équipes sont gérées via le H2H.
- Les mini-classements à 3 équipes ou plus restent simplifiés.
'''
)