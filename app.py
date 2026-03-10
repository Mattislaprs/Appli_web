import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import unicodedata
import streamlit as st

st.set_page_config(page_title="Analyse tactique", layout="wide")


# =========================
# Utilitaires
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uniformise les noms de colonnes fréquents."""
    renamed = {}

    for col in df.columns:
        col_clean = str(col).strip()

        col_no_accents = ''.join(
            c for c in unicodedata.normalize("NFD", col_clean)
            if unicodedata.category(c) != "Mn"
        )

        col_lower = col_no_accents.lower()

        if col_lower == "row":
            renamed[col] = "Row"
        elif col_lower == "issue":
            renamed[col] = "Issue"
        elif col_lower == "joueur":
            renamed[col] = "Joueur"
        elif col_lower == "situation":
            renamed[col] = "Situation"
        elif col_lower == "choix":
            renamed[col] = "Choix"
        elif col_lower == "choix duel":
            renamed[col] = "Choix duel"
        elif col_lower == "hauteur de bloc":
            renamed[col] = "Hauteur de bloc"
        elif col_lower == "phase de jeu":
            renamed[col] = "Phase de jeu"
        elif col_lower == "rapport numerique":
            renamed[col] = "Rapport numerique"
        else:
            renamed[col] = col_clean

    return df.rename(columns=renamed)


def ensure_row_column(df: pd.DataFrame) -> pd.DataFrame:
    """Crée une colonne Row si elle n'existe pas encore."""
    if "Row" in df.columns:
        return df

    if "Situation" in df.columns:
        df["Row"] = df["Situation"].astype(str)
        return df

    df["Row"] = [f"Situation {i}" for i in range(len(df))]
    return df


def to_float(value):
    """Convertit une valeur en float, y compris avec virgule décimale."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if value == "":
        return None

    value = value.replace(" ", "").replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def convert_coords(x, y, source_scale: str):
    """Convertit les coordonnées vers un terrain 120 x 60."""
    if x is None or y is None:
        return None, None

    if source_scale == "100 x 100":
        return x * 120 / 100, y * 60 / 100

    if source_scale == "120 x 120":
        return x, y * 60 / 120

    if source_scale == "60 x 120":
        return x * 120 / 60, y * 60 / 120

    # 120 x 60
    return x, y


def clamp_coords(x, y):
    """Ramène un point à l'intérieur du terrain."""
    x = max(0, min(120, x))
    y = max(0, min(60, y))
    return x, y


def draw_pitch(ax):
    """Dessine un terrain 120 x 60."""
    ax.set_facecolor("#1f7a3d")

    ax.plot([0, 120], [0, 0], color="white", lw=2)
    ax.plot([0, 120], [60, 60], color="white", lw=2)
    ax.plot([0, 0], [0, 60], color="white", lw=2)
    ax.plot([120, 120], [0, 60], color="white", lw=2)

    ax.plot([60, 60], [0, 60], color="white", lw=2)

    centre = Circle((60, 30), 9.15, fill=False, color="white", lw=2)
    ax.add_patch(centre)
    ax.scatter(60, 30, color="white", s=15)

    ax.add_patch(Rectangle((0, 18), 18, 24, fill=False, edgecolor="white", lw=2))
    ax.add_patch(Rectangle((102, 18), 18, 24, fill=False, edgecolor="white", lw=2))

    ax.add_patch(Rectangle((0, 24), 6, 12, fill=False, edgecolor="white", lw=2))
    ax.add_patch(Rectangle((114, 24), 6, 12, fill=False, edgecolor="white", lw=2))

    ax.scatter(12, 30, color="white", s=15)
    ax.scatter(108, 30, color="white", s=15)

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
    adv_count = 0

    for col in df.columns:
        if col.startswith("X_"):
            suffix = col[2:]
            y_col = f"Y_{suffix}"

            if y_col in df.columns:
                label = suffix.upper()

                if label in ["ADV", "DLADV"]:
                    adv_count += 1
                    label = f"ADV{adv_count}"

                players[label] = (col, y_col)

    return dict(sorted(players.items()))


ROLE_COLORS = {
    "ADV1": "red",
    "ADV2": "red",
    "BAL": "white",
    "DC": "blue",
    "MDC": "blue",
    "MC": "blue",
    "AL": "blue",
    "LAT": "yellow"
}


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

    if label == "LAT" and "Joueur" in line.index and pd.notna(line["Joueur"]) and str(line["Joueur"]).strip() != "":
        return str(line["Joueur"])

    return None


def get_select_filter_options(df: pd.DataFrame, column_name: str, all_label: str = "Toutes"):
    if column_name not in df.columns:
        return [all_label]

    values = (
        df[column_name]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )

    unique_values = sorted(values.unique().tolist())
    return [all_label] + unique_values


def plot_situation(
    df: pd.DataFrame,
    row_index: int,
    player_map: dict,
    source_scale: str,
    selected_players: list[str],
    clamp_outside: bool,
    point_size: int,
):
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
            debug_rows.append({
                "Joueur tactique": label,
                "Joueur réel": player_name,
                "X brut": x_raw,
                "Y brut": y_raw,
                "Statut": status,
            })
            continue

        x, y = convert_coords(x_num, y_num, source_scale)

        if x is None or y is None:
            status = "conversion impossible"
            debug_rows.append({
                "Joueur tactique": label,
                "Joueur réel": player_name,
                "X brut": x_raw,
                "Y brut": y_raw,
                "Statut": status,
            })
            continue

        if not (0 <= x <= 120 and 0 <= y <= 60):
            if clamp_outside:
                x, y = clamp_coords(x, y)
                status = "hors terrain puis recalé"
            else:
                status = f"hors terrain ({x:.2f}, {y:.2f})"
                debug_rows.append({
                    "Joueur tactique": label,
                    "Joueur réel": player_name,
                    "X brut": x_raw,
                    "Y brut": y_raw,
                    "Statut": status,
                })
                continue

        color = ROLE_COLORS.get(label, "white")
        ax.scatter(
            x,
            y,
            s=point_size,
            color=color,
            edgecolors="black",
            linewidths=2,
            zorder=4,
        )

        display_text = label if player_name is None else f"{label} - {player_name}"
        ax.text(x + 1.2, y + 0.6, display_text, color="white", fontsize=10, weight="bold", zorder=5)

        plotted.append({
            "Joueur tactique": label,
            "Joueur réel": player_name,
            "X terrain": round(x, 2),
            "Y terrain": round(y, 2),
            "X brut": x_raw,
            "Y brut": y_raw,
            "Statut": status,
        })

        debug_rows.append({
            "Joueur tactique": label,
            "Joueur réel": player_name,
            "X brut": x_raw,
            "Y brut": y_raw,
            "Statut": status,
        })

    title_issue = line.get("Issue", "Sans Issue") if "Issue" in df.columns else "Sans Issue"
    title_row = line.get("Row", "Sans Situation") if "Row" in df.columns else "Sans Situation"

    ax.set_title(
        f"Issue : {title_issue} | Situation : {title_row} | Ligne : {row_index}",
        color="white",
        fontsize=14,
        weight="bold",
    )

    return fig, pd.DataFrame(plotted), pd.DataFrame(debug_rows)


def overlay_average_positions(
    ax,
    df_avg: pd.DataFrame,
    player_map: dict,
    source_scale: str,
    selected_players: list[str],
    clamp_outside: bool,
    point_size: int,
):
    """Ajoute les positions moyennes sur le graphique existant."""
    average_rows = []

    for label, (x_col, y_col) in player_map.items():
        if selected_players and label not in selected_players:
            continue

        if x_col not in df_avg.columns or y_col not in df_avg.columns:
            continue

        x_vals = df_avg[x_col].apply(to_float).dropna()
        y_vals = df_avg[y_col].apply(to_float).dropna()

        if len(x_vals) == 0 or len(y_vals) == 0:
            average_rows.append({
                "Joueur tactique": label,
                "X moyen": None,
                "Y moyen": None,
                "Nb points": 0,
                "Statut": "aucune coordonnée exploitable",
            })
            continue

        x_mean = x_vals.mean()
        y_mean = y_vals.mean()

        x, y = convert_coords(x_mean, y_mean, source_scale)

        status = "ok"
        if x is None or y is None:
            average_rows.append({
                "Joueur tactique": label,
                "X moyen": None,
                "Y moyen": None,
                "Nb points": len(x_vals),
                "Statut": "conversion impossible",
            })
            continue

        if not (0 <= x <= 120 and 0 <= y <= 60):
            if clamp_outside:
                x, y = clamp_coords(x, y)
                status = "hors terrain puis recalé"
            else:
                average_rows.append({
                    "Joueur tactique": label,
                    "X moyen": round(x, 2),
                    "Y moyen": round(y, 2),
                    "Nb points": len(x_vals),
                    "Statut": f"hors terrain ({x:.2f}, {y:.2f})",
                })
                continue

        color = ROLE_COLORS.get(label, "white")

        ax.scatter(
            x,
            y,
            s=point_size + 140,
            color=color,
            alpha=0.35,
            edgecolors="white",
            linewidths=2.5,
            zorder=2,
        )

        ax.scatter(
            x,
            y,
            s=35,
            color="white",
            edgecolors="black",
            linewidths=1,
            zorder=6,
        )

        ax.text(
            x + 1.2,
            y - 1.2,
            f"{label} moy",
            color="white",
            fontsize=9,
            weight="bold",
            zorder=6,
        )

        average_rows.append({
            "Joueur tactique": label,
            "X moyen": round(x, 2),
            "Y moyen": round(y, 2),
            "Nb points": len(x_vals),
            "Statut": status,
        })

    return pd.DataFrame(average_rows)


# =========================
# Interface
# =========================
st.title("Mini appli d'analyse tactique")
st.caption("Filtres par Issue, Joueur, Situation, Choix, Choix duel, Hauteur de bloc, Phase de jeu et Rapport numerique.")

uploaded_file = st.file_uploader("Dépose ton CSV", type=["csv"])

if uploaded_file is None:
    st.info("Charge un CSV pour commencer.")
    st.stop()

# Lecture du fichier
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
df = ensure_row_column(df)
df = df.reset_index(drop=True)

player_map = build_player_map(df)

if not player_map:
    st.error("Aucune paire de colonnes X_/Y_ détectée dans le fichier.")
    st.stop()

with st.sidebar:
    st.header("Filtres")

    source_scale = st.radio(
        "Échelle des coordonnées source",
        ["100 x 100", "120 x 120", "60 x 120", "120 x 60"],
        index=0,
        help="Choisis l'échelle d'origine des coordonnées. Elles seront converties vers un terrain 120 x 60.",
    )

    clamp_outside = st.checkbox(
        "Forcer les joueurs hors terrain à apparaître sur le bord",
        value=True,
    )

    point_size = st.slider(
        "Taille des joueurs",
        min_value=50,
        max_value=400,
        value=180,
        step=10,
    )

    display_mode = st.radio(
        "Mode d'affichage",
        ["Situation actuelle", "Situation + moyenne", "Moyenne seule"],
        index=0,
    )

    issue_options = get_select_filter_options(df, "Issue")
    selected_issue = st.selectbox("Issue", issue_options)

    if "Joueur" in df.columns:
        real_player_options = ["Tous"] + sorted(
            df["Joueur"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        selected_real_player = st.selectbox("Joueur", real_player_options)
    else:
        selected_real_player = "Tous"

    situation_options = get_select_filter_options(df, "Situation")
    selected_situation_filter = st.selectbox("Situation", situation_options)

    choix_options = get_select_filter_options(df, "Choix")
    selected_choix = st.selectbox("Choix", choix_options)

    choix_duel_options = get_select_filter_options(df, "Choix duel")
    selected_choix_duel = st.selectbox("Choix duel", choix_duel_options)

    hauteur_bloc_options = get_select_filter_options(df, "Hauteur de bloc")
    selected_hauteur_bloc = st.selectbox("Hauteur de bloc", hauteur_bloc_options)

    phase_jeu_options = get_select_filter_options(df, "Phase de jeu")
    selected_phase_jeu = st.selectbox("Phase de jeu", phase_jeu_options)

    rapport_numerique_options = get_select_filter_options(df, "Rapport numerique")
    selected_rapport_numerique = st.selectbox("Rapport numerique", rapport_numerique_options)

    available_players = list(player_map.keys())
    selected_players = st.multiselect(
        "Points tactiques à afficher",
        available_players,
        default=available_players,
    )

# =========================
# Filtrage
# =========================
filtered_df = df.copy()

if selected_issue != "Toutes" and "Issue" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Issue"].astype(str).str.strip() == selected_issue].copy()

if selected_real_player != "Tous" and "Joueur" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Joueur"].astype(str).str.strip() == selected_real_player].copy()

if selected_situation_filter != "Toutes" and "Situation" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Situation"].astype(str).str.strip() == selected_situation_filter].copy()

if selected_choix != "Toutes" and "Choix" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Choix"].astype(str).str.strip() == selected_choix].copy()

if selected_choix_duel != "Toutes" and "Choix duel" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Choix duel"].astype(str).str.strip() == selected_choix_duel].copy()

if selected_hauteur_bloc != "Toutes" and "Hauteur de bloc" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Hauteur de bloc"].astype(str).str.strip() == selected_hauteur_bloc].copy()

if selected_phase_jeu != "Toutes" and "Phase de jeu" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Phase de jeu"].astype(str).str.strip() == selected_phase_jeu].copy()

if selected_rapport_numerique != "Toutes" and "Rapport numerique" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["Rapport numerique"].astype(str).str.strip() == selected_rapport_numerique
    ].copy()

if filtered_df.empty:
    st.warning("Aucune ligne ne correspond au filtre choisi.")
    st.stop()

filtered_df["display_option"] = filtered_df.apply(format_option, axis=1)
options = filtered_df["display_option"].tolist()
selected_option = st.selectbox("Situation", options)
selected_index = int(selected_option.split(" - ")[0])

col1, col2 = st.columns([2.1, 1])

with col1:
    average_df = pd.DataFrame()

    if display_mode == "Situation actuelle":
        fig, plotted_df, debug_df = plot_situation(
            df=df,
            row_index=selected_index,
            player_map=player_map,
            source_scale=source_scale,
            selected_players=selected_players,
            clamp_outside=clamp_outside,
            point_size=point_size,
        )

    elif display_mode == "Situation + moyenne":
        fig, plotted_df, debug_df = plot_situation(
            df=df,
            row_index=selected_index,
            player_map=player_map,
            source_scale=source_scale,
            selected_players=selected_players,
            clamp_outside=clamp_outside,
            point_size=point_size,
        )

        average_df = overlay_average_positions(
            ax=fig.axes[0],
            df_avg=filtered_df,
            player_map=player_map,
            source_scale=source_scale,
            selected_players=selected_players,
            clamp_outside=clamp_outside,
            point_size=point_size,
        )

    else:  # Moyenne seule
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_pitch(ax)

        plotted_df = pd.DataFrame()
        debug_df = pd.DataFrame()

        average_df = overlay_average_positions(
            ax=ax,
            df_avg=filtered_df,
            player_map=player_map,
            source_scale=source_scale,
            selected_players=selected_players,
            clamp_outside=clamp_outside,
            point_size=point_size,
        )

        ax.set_title(
            "Positions moyennes",
            color="white",
            fontsize=14,
            weight="bold",
        )

    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Résumé")
    current_line = df.loc[selected_index]

    st.write(f"**Ligne :** {selected_index}")
    st.write(f"**Situation :** {current_line.get('Row', 'NA')}")

    if "Issue" in df.columns:
        st.write(f"**Issue :** {current_line.get('Issue', 'NA')}")
    if "Joueur" in df.columns:
        st.write(f"**Joueur :** {current_line.get('Joueur', 'NA')}")
    if "Choix" in df.columns:
        st.write(f"**Choix :** {current_line.get('Choix', 'NA')}")
    if "Choix duel" in df.columns:
        st.write(f"**Choix duel :** {current_line.get('Choix duel', 'NA')}")
    if "Hauteur de bloc" in df.columns:
        st.write(f"**Hauteur de bloc :** {current_line.get('Hauteur de bloc', 'NA')}")
    if "Phase de jeu" in df.columns:
        st.write(f"**Phase de jeu :** {current_line.get('Phase de jeu', 'NA')}")
    if "Rapport numerique" in df.columns:
        st.write(f"**Rapport numerique :** {current_line.get('Rapport numerique', 'NA')}")

    st.write(f"**Mode :** {display_mode}")

    if display_mode == "Moyenne seule":
        st.write(f"**Joueurs moyens affichés :** {len(average_df)}")
    else:
        st.write(f"**Joueurs affichés :** {len(plotted_df)}")

    if display_mode in ["Situation actuelle", "Situation + moyenne"]:
        with st.expander("Coordonnées de la situation affichée", expanded=True):
            if plotted_df.empty:
                st.write("Aucun joueur affiché.")
            else:
                st.dataframe(plotted_df, use_container_width=True)

    if display_mode in ["Situation + moyenne", "Moyenne seule"]:
        with st.expander("Positions moyennes", expanded=True):
            if average_df.empty:
                st.write("Aucune moyenne affichée.")
            else:
                st.dataframe(average_df, use_container_width=True)

    if display_mode != "Moyenne seule":
        with st.expander("Diagnostic complet"):
            st.dataframe(debug_df, use_container_width=True)

st.divider()

with st.expander("Aperçu des données filtrées"):
    st.dataframe(filtered_df.drop(columns=["display_option"]), use_container_width=True)
