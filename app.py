import io
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import streamlit as st

st.set_page_config(page_title="Analyse tactique", layout="wide")


# =========================
# Utilitaires
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uniformise quelques noms de colonnes fréquents."""
    renamed = {}
    for col in df.columns:
        if col == "ROW":
            renamed[col] = "Row"
    if renamed:
        df = df.rename(columns=renamed)
    return df


def to_float(value):
    """Convertit proprement une valeur en float, y compris avec virgule décimale."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if value == "":
        return None

    # Cas fréquent : '26,2'
    # Cas ambigu : '117,101' -> on l'interprète comme 117.101
    value = value.replace(" ", "").replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def convert_coords(x, y, source_scale: str):
    """Convertit les coordonnées vers un terrain 120 x 60."""
    if x is None or y is None:
        return None, None

    # données normalisées
    if source_scale == "100 x 100":
        return x * 120 / 100, y * 60 / 100

    # données 120 x 120 (souvent export tracking)
    if source_scale == "120 x 120":
        return x, y * 60 / 120

    # déjà au bon format
    return x, y


def clamp_coords(x, y):
    """Ramène un point à l'intérieur du terrain si nécessaire."""
    x = max(0, min(120, x))
    y = max(0, min(60, y))
    return x, y


def draw_pitch(ax):
    """Dessine un terrain 120 x 60."""
    # Fond
    ax.set_facecolor("#1f7a3d")

    # Contours
    ax.plot([0, 120], [0, 0], color="white", lw=2)
    ax.plot([0, 120], [60, 60], color="white", lw=2)
    ax.plot([0, 0], [0, 60], color="white", lw=2)
    ax.plot([120, 120], [0, 60], color="white", lw=2)

    # Ligne médiane
    ax.plot([60, 60], [0, 60], color="white", lw=2)

    # Cercle central
    centre = Circle((60, 30), 9.15, fill=False, color="white", lw=2)
    ax.add_patch(centre)
    ax.scatter(60, 30, color="white", s=15)

    # Surfaces de réparation
    ax.add_patch(Rectangle((0, 18), 18, 24, fill=False, edgecolor="white", lw=2))
    ax.add_patch(Rectangle((102, 18), 18, 24, fill=False, edgecolor="white", lw=2))

    # Surfaces de but
    ax.add_patch(Rectangle((0, 24), 6, 12, fill=False, edgecolor="white", lw=2))
    ax.add_patch(Rectangle((114, 24), 6, 12, fill=False, edgecolor="white", lw=2))

    # Points de penalty
    ax.scatter(12, 30, color="white", s=15)
    ax.scatter(108, 30, color="white", s=15)

    # Arcs de penalty
    ax.add_patch(Arc((12, 30), 18.3, 18.3, angle=0, theta1=310, theta2=50, color="white", lw=2))
    ax.add_patch(Arc((108, 30), 18.3, 18.3, angle=0, theta1=130, theta2=230, color="white", lw=2))

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def build_player_map(df: pd.DataFrame):
    """Détecte automatiquement les paires X_/Y_ et crée un mapping joueur -> colonnes."""
    players = {}
    for col in df.columns:
        if col.startswith("X_"):
            suffix = col[2:]
            y_col = f"Y_{suffix}"
            if y_col in df.columns:
                players[suffix.upper()] = (col, y_col)
    return dict(sorted(players.items()))


def format_option(row) -> str:
    issue = str(row.get("Issue", "Sans Issue")) if not pd.isna(row.get("Issue", None)) else "Sans Issue"
    situation = str(row.get("Row", "Sans Situation")) if not pd.isna(row.get("Row", None)) else "Sans Situation"
    return f"{row.name} - {issue} - {situation}"


def get_player_name_for_label(line: pd.Series, label: str):
    """Essaie de retrouver le nom réel du joueur associé à un point tactique."""
    candidates = [
        f"Joueur_{label.lower()}",
        f"Joueur_{label}",
        f"PLAYER_{label.lower()}",
        f"PLAYER_{label}",
        f"Nom_{label.lower()}",
        f"Nom_{label}",
    ]

    for col in candidates:
        if col in line.index and pd.notna(line[col]) and str(line[col]).strip() != "":
            return str(line[col])

    # Cas demandé : utiliser la colonne 'Joueur' pour le point LAT
    if label == "LAT" and "Joueur" in line.index and pd.notna(line["Joueur"]) and str(line["Joueur"]).strip() != "":
        return str(line["Joueur"])

    return None


def plot_situation(df: pd.DataFrame, row_index: int, player_map: dict, source_scale: str, selected_players: list[str], clamp_outside: bool):
    line = df.loc[row_index]

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_pitch(ax)

    plotted = []
    debug_rows = []

    for label, (x_col, y_col) in player_map.items():
        if selected_players and label not in selected_players:
            continue

        x_raw = line.get(x_col)
        y_raw = line.get(y_col)
        x_num = to_float(x_raw)
        y_num = to_float(y_raw)
        player_name = get_player_name_for_label(line, label)

        status = "ok"
        if x_num is None or y_num is None:
            status = "valeur manquante / non convertible"
            debug_rows.append({"Joueur tactique": label, "Joueur réel": player_name, "X brut": x_raw, "Y brut": y_raw, "Statut": status})
            continue

        x, y = convert_coords(x_num, y_num, source_scale)

        if x is None or y is None:
            status = "conversion impossible"
            debug_rows.append({"Joueur tactique": label, "Joueur réel": player_name, "X brut": x_raw, "Y brut": y_raw, "Statut": status})
            continue

        if not (0 <= x <= 120 and 0 <= y <= 60):
            if clamp_outside:
                x, y = clamp_coords(x, y)
                status = "hors terrain puis recalé"
            else:
                status = f"hors terrain ({x:.2f}, {y:.2f})"
                debug_rows.append({"Joueur tactique": label, "Joueur réel": player_name, "X brut": x_raw, "Y brut": y_raw, "Statut": status})
                continue

        ax.scatter(x, y, s=180, edgecolors="black", linewidths=1.2)
        display_text = label if player_name is None else f"{label} - {player_name}"
        ax.text(x + 1.2, y + 0.6, display_text, color="white", fontsize=10, weight="bold")
        plotted.append({
            "Joueur tactique": label,
            "Joueur réel": player_name,
            "X terrain": round(x, 2),
            "Y terrain": round(y, 2),
            "X brut": x_raw,
            "Y brut": y_raw,
            "Statut": status,
        })
        debug_rows.append({"Joueur tactique": label, "Joueur réel": player_name, "X brut": x_raw, "Y brut": y_raw, "Statut": status})

    title_issue = line.get("Issue", "Sans Issue") if "Issue" in df.columns else "Sans Issue"
    title_row = line.get("Row", "Sans Situation") if "Row" in df.columns else "Sans Situation"
    ax.set_title(f"Issue : {title_issue} | Situation : {title_row} | Ligne : {row_index}", color="white", fontsize=14, weight="bold")

    return fig, pd.DataFrame(plotted), pd.DataFrame(debug_rows)


# =========================
# Interface
# =========================
st.title("Mini appli d'analyse tactique")
st.caption("Filtres par Issue, Joueur et Situation, avec affichage sur terrain 120 x 60.")

uploaded_file = st.file_uploader("Dépose ton CSV", type=["csv"])

if uploaded_file is None:
    st.info("Charge un CSV pour commencer.")
    st.stop()

# Lecture du fichier
# Lecture du fichier (détection automatique du séparateur ";" ou ",")
try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
except Exception as e:
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=";")
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=",")
        except Exception:
            st.error(f"Impossible de lire le fichier CSV : {e}")
            st.stop()

df = normalize_columns(df)
df = df.reset_index(drop=True)

if "Row" not in df.columns:
    st.error("La colonne 'Row' est introuvable dans le fichier. Vérifie le CSV.")
    st.stop()

player_map = build_player_map(df)

if not player_map:
    st.error("Aucune paire de colonnes X_/Y_ détectée dans le fichier.")
    st.stop()

with st.sidebar:
    st.header("Filtres")

    source_scale = st.radio(
        "Échelle des coordonnées source",
        ["100 x 100", "120 x 120", "120 x 60"],
        index=0,
        help="Choisis l'échelle d'origine des coordonnées. Elles seront converties vers un terrain 120 x 60."
    )

    clamp_outside = st.checkbox(
        "Forcer les joueurs hors terrain à apparaître sur le bord",
        value=True
    )

    if "Issue" in df.columns:
        issue_options = ["Toutes"] + sorted(df["Issue"].dropna().astype(str).unique().tolist())
        selected_issue = st.selectbox("Issue", issue_options)
    else:
        selected_issue = "Toutes"

    # Filtre par joueur réel (ex: colonne 'Joueur')
    if "Joueur" in df.columns:
        real_player_options = ["Tous"] + sorted(df["Joueur"].dropna().astype(str).unique().tolist())
        selected_real_player = st.selectbox("Joueur", real_player_options)
    else:
        selected_real_player = "Tous"

    # Nouveau filtre par colonne Situation
    if "Situation" in df.columns:
        situation_options = ["Toutes"] + sorted(df["Situation"].dropna().astype(str).unique().tolist())
        selected_situation_filter = st.selectbox("Situation", situation_options)
    else:
        selected_situation_filter = "Toutes"

    available_players = list(player_map.keys())
    selected_players = st.multiselect(
        "Points tactiques à afficher",
        available_players,
        default=available_players
    )

# Filtre par Issue
filtered_df = df.copy()
if selected_issue != "Toutes":
    filtered_df = filtered_df[filtered_df["Issue"].astype(str) == selected_issue].copy()

if selected_real_player != "Tous" and "Joueur" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Joueur"].astype(str) == selected_real_player].copy()

# Filtre Situation
if 'selected_situation_filter' in locals() and selected_situation_filter != "Toutes" and "Situation" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Situation"].astype(str) == selected_situation_filter].copy()

if filtered_df.empty:
    st.warning("Aucune ligne ne correspond au filtre choisi.")
    st.stop()

filtered_df["display_option"] = filtered_df.apply(format_option, axis=1)
options = filtered_df["display_option"].tolist()
selected_option = st.selectbox("Situation", options)
selected_index = int(selected_option.split(" - ")[0])

col1, col2 = st.columns([2.1, 1])

with col1:
    fig, plotted_df, debug_df = plot_situation(
        df=df,
        row_index=selected_index,
        player_map=player_map,
        source_scale=source_scale,
        selected_players=selected_players,
        clamp_outside=clamp_outside,
    )
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Résumé")
    current_line = df.loc[selected_index]
    st.write(f"**Ligne :** {selected_index}")
    st.write(f"**Situation :** {current_line.get('Row', 'NA')}")
    if "Issue" in df.columns:
        st.write(f"**Issue :** {current_line.get('Issue', 'NA')}")
    st.write(f"**Joueurs affichés :** {len(plotted_df)}")

    with st.expander("Coordonnées affichées", expanded=True):
        if plotted_df.empty:
            st.write("Aucun joueur affiché.")
        else:
            st.dataframe(plotted_df, use_container_width=True)

    with st.expander("Diagnostic complet"):
        st.dataframe(debug_df, use_container_width=True)

st.divider()

with st.expander("Aperçu des données filtrées"):
    st.dataframe(filtered_df.drop(columns=["display_option"]), use_container_width=True)
