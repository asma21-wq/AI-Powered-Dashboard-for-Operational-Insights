import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from io import BytesIO

# Page setup
st.set_page_config(page_title="Dashboard Élagage", layout="wide")
st.title("Dashboard Élagage")
dashboard_insights = []

# Color palette
DARK_orange = '#e27602'
LIGHT_orange = '#f5c77e'
LIGHT_BLUE = '#2699e6'

# ➤ Convert DataFrame to image
def dataframe_to_image(df, title=None):
    fig, ax = plt.subplots(figsize=(min(12, len(df.columns) * 2), min(0.5 * len(df), 12)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    if title:
        plt.title(title, fontsize=12)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# ➤ Display chart + downloads with centered layout
def display_chart_and_downloads(fig, data_df, chart_filename, csv_filename, key_prefix=""):
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("📅 Télécharger le graphique", data=buf.getvalue(), file_name=chart_filename, mime="image/png", key=f"{key_prefix}_img")
    csv = data_df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Télécharger les données CSV", data=csv, file_name=csv_filename, mime="text/csv", key=f"{key_prefix}_csv")


def summarize_for_llm(insights: list):
    summary = "## 🔍 Résumé du Dashboard - Analyse Opérationnelle\n"
    summary += "\n".join(f"- {line}" for line in insights)
    return summary


from langchain.llms import LlamaCpp

def generate_llm_report(summary_text):
    llm = LlamaCpp(
        model_path=r"C:\Users\mhatl\llama.cpp\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Adjust path
        temperature=0.2,
        max_tokens=1024,
        top_p=0.95,
        n_ctx=2048,
        verbose=False
    )

    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
Vous êtes un conseiller stratégique. Voici un résumé d’un tableau de bord opérationnel :

---
{summary}
---

🎯 Objectifs :
1. Résumez les principales tendances.
2. Donnez 3 recommandations concrètes pour le manager.
3. Soulignez toute anomalie ou point d'attention.

📝 Le ton doit être professionnel et synthétique.
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"summary": summary_text})

# Uploader
uploaded_file = st.file_uploader("Charger le fichier Excel", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    # Récupérer min/max dates sur "Nouvelle instance" pour filtre global
    df_dates = pd.read_excel(xls, sheet_name="Nouvelle instance")
    df_dates.columns = df_dates.columns.str.strip()
    df_dates["date d'intégration"] = pd.to_datetime(df_dates["date d'intégration"], errors="coerce")
    min_date = df_dates["date d'intégration"].min()
    max_date = df_dates["date d'intégration"].max()

    # Sélection plage de dates
    date_range = st.date_input("Filtrer la période (date d'intégration)", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

        # Menu d'affichage
        chart_option = st.selectbox("Que voulez-vous afficher ?", [
            "Rien",
            "Afficher tout",
            "Flux entrant par semaine (Abandon doublon / Demande validé)",
            "Répartition des 'en attente complément' par région",
            "Taux de conformité par région",
            "Dossiers clôturés par semaine (élagage réalisé)",
          "Relance 1 Mail envoyé à la maire par semaine"
        ])
    def plot_option1():
        df = pd.read_excel(xls, sheet_name="Nouvelle instance")
        df.columns = df.columns.str.strip()
        df["date d'intégration"] = pd.to_datetime(df["date d'intégration"], errors="coerce")
        df["Action"] = df["Action"].astype(str).str.strip()
        df["Num Semaine"] = pd.to_numeric(df["Num Semaine"], errors="coerce")
        df["N° dossier"] = df["N° dossier"].astype(str).str.strip()

        df_filtered = df[
            (df["date d'intégration"] >= start_date) &
            (df["date d'intégration"] <= end_date) &
            (df["Action"].isin(["Abandon doublon", "Demande validé", "en attente complément"])) &
            df["Num Semaine"].notna()
        ]

        not_zero_mask = df_filtered["N° dossier"] != "0"
        duplicated_mask = df_filtered[not_zero_mask].duplicated(subset=["Action", "N° dossier"], keep="first")
        to_remove_indices = df_filtered[not_zero_mask].loc[duplicated_mask].index
        removed_duplicates = df_filtered.loc[to_remove_indices]
        df_filtered = df_filtered.drop(index=to_remove_indices)

        nb_doublons = len(removed_duplicates)
        if nb_doublons > 0:
            st.warning(f"⚠️ {nb_doublons} doublon(s) supprimé(s).")
            st.subheader("🔍 Détails des doublons supprimés")
            st.dataframe(removed_duplicates.reset_index(drop=True))
        else:
            st.success("✅ Aucun doublon détecté.")

        grouped = df_filtered.groupby(["Num Semaine", "Action"]).size().unstack(fill_value=0).sort_index()
        colors = {
            "Demande validé": DARK_orange,
            "Abandon doublon": LIGHT_orange,
            "en attente complément": "#b4d7f6"
        }

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = grouped.plot(
            kind="bar",
            stacked=True,
            color=[colors.get(col, "#cccccc") for col in grouped.columns],
            edgecolor="black",
            ax=ax
        )

        for container in bars.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        str(int(height)),
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='white',
                        fontweight='bold'
                    )

        ax.set_title("Flux entrant par semaine", fontsize=14)
        ax.set_xlabel("Numéro de Semaine")
        ax.set_ylabel("Nombre de demandes")
        ax.legend(title="Type d'action")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=0)
        plt.tight_layout()

        df_out = grouped.reset_index()
        display_chart_and_downloads(fig, df_out, "flux_entrant.png", "flux_entrant.csv", key_prefix="option1")
        if "Demande validé" in grouped.columns:
            week_max = grouped["Demande validé"].idxmax()
            val_max = grouped["Demande validé"].max()
            dashboard_insights.append(
                f"📊 [Flux entrant] : Semaine {week_max} a eu le plus de 'Demande validé' ({val_max} dossiers)."
            )


    def plot_option2():
        df = pd.read_excel(xls, sheet_name="Nouvelle instance")
        df.columns = df.columns.str.strip()
        df["date d'intégration"] = pd.to_datetime(df["date d'intégration"], errors="coerce")

        df["Action"] = df["Action"].astype(str).str.strip().str.lower()
        df["N° dossier"] = df["N° dossier"].astype(str).str.strip()

        # Filter by date range
        df_filtered = df[(df["date d'intégration"] >= start_date) & (df["date d'intégration"] <= end_date)]

        not_zero_mask = df_filtered["N° dossier"] != "0"
        duplicated_mask = df_filtered[not_zero_mask].duplicated(subset=["Action", "N° dossier"], keep="first")
        to_remove_indices = df_filtered[not_zero_mask].loc[duplicated_mask].index
        removed_duplicates = df_filtered.loc[to_remove_indices]
        df_filtered = df_filtered.drop(index=to_remove_indices)

        nb_doublons = len(removed_duplicates)
        if nb_doublons > 0:
            st.warning(f"⚠️ {nb_doublons} doublon(s) supprimé(s).")
            st.subheader("🔍 Détails des doublons supprimés")
            st.dataframe(removed_duplicates.reset_index(drop=True))
        else:
            st.success("✅ Aucun doublon détecté.")

        total = df_filtered.groupby("Région").size().reset_index(name="Total")
        waiting = df_filtered[df_filtered["Action"] == "en attente complément"].groupby("Région").size().reset_index(name="En attente")
        merged = pd.merge(waiting, total, on="Région", how="left")
        merged["Attente/Total"] = merged["En attente"].astype(str) + " / " + merged["Total"].astype(str)

        st.subheader("Répartition des 'en attente complément'")
        st.dataframe(merged[["Région", "Attente/Total"]])

        # Check if merged DataFrame is empty before exporting image and CSV
        if not merged.empty:
            table_png = dataframe_to_image(merged[["Région", "Attente/Total"]])
            st.download_button(
                "🖼️ Télécharger le tableau (PNG)", data=table_png,
                file_name="attente_complement_tableau.png", mime="image/png"
            )

            csv = merged[["Région", "Attente/Total"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Télécharger le tableau CSV", data=csv,
                file_name="attente_complement.csv", mime="text/csv", key="option2_csv"
            )
        else:
            st.info("Aucune donnée disponible pour générer le tableau.")
        if not merged.empty:
            top_region = merged.loc[merged["En attente"].idxmax(), "Région"]
            dashboard_insights.append(
                f"📌 [Attente complément] : Région avec le plus de cas en attente : {top_region} ({merged['En attente'].max()} cas)."
            )


    def plot_option3():
        df = pd.read_excel(xls, sheet_name="Nouvelle instance")
        df.columns = df.columns.str.strip()
        df["date d'intégration"] = pd.to_datetime(df["date d'intégration"], errors="coerce")

        df["Action"] = df["Action"].astype(str).str.strip().str.lower()
        df["N° dossier"] = df["N° dossier"].astype(str).str.strip()
        df_filtered = df[(df["date d'intégration"] >= start_date) & (df["date d'intégration"] <= end_date)]

        not_zero_mask = df_filtered["N° dossier"] != "0"
        duplicated_mask = df_filtered[not_zero_mask].duplicated(subset=["Action", "N° dossier"], keep="first")
        to_remove_indices = df_filtered[not_zero_mask].loc[duplicated_mask].index
        removed_duplicates = df_filtered.loc[to_remove_indices]
        df = df_filtered.drop(index=to_remove_indices)

        nb_doublons = len(removed_duplicates)
        if nb_doublons > 0:
            st.warning(f"⚠️ {nb_doublons} doublon(s) supprimé(s).")
            st.subheader("🔍 Détails des doublons supprimés")
            st.dataframe(removed_duplicates.reset_index(drop=True))
        else:
            st.success("✅ Aucun doublon détecté.")


        actions_valides = ["demande validé", "abandon doublon"]
        total = df_filtered.groupby("Région").size()
        conformes = df_filtered[df_filtered["Action"].isin(actions_valides)].groupby("Région").size()
        taux = (conformes / total).fillna(0) * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(taux.index, taux.values, color=LIGHT_BLUE)
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 1, f"{y:.1f}%", ha="center", fontsize=9)

        ax.set_title("Taux de conformité par région", fontsize=14)
        ax.set_ylabel("Taux de conformité (%)")
        ax.set_ylim(0, 110)

        plt.xticks(rotation=0)
        plt.tight_layout()

        taux_df = taux.reset_index()
        taux_df.columns = ["Région", "Taux de conformité (%)"]
        display_chart_and_downloads(fig, taux_df, "taux_conformite.png", "taux_conformite.csv", key_prefix="option3")
        if not taux_df.empty:
            best_region = taux_df.iloc[taux_df["Taux de conformité (%)"].idxmax()]
            dashboard_insights.append(
                f"✅ [Conformité] : {best_region['Région']} a la meilleure conformité avec {best_region['Taux de conformité (%)']:.1f}%."
            )


    def plot_option4():
        df = pd.read_excel(xls, sheet_name="Traitement Dossier")
        df.columns = df.columns.str.strip()
        df["Heure de fin"] = pd.to_datetime(df["Heure de fin"], errors="coerce")

        df["Action"] = df["Action"].astype(str).str.strip()
        df["N° Dossier"] = df["N° Dossier"].astype(str).str.strip()

        # Filter rows with specific action and date range
        df = df[
            (df["Action"] == "Dossier clôturé élagage réalisé") &
            (df["Heure de fin"] >= start_date) &
            (df["Heure de fin"] <= end_date)
        ]

        not_zero_mask = df["N° Dossier"] != "0"
        duplicated_mask = df[not_zero_mask].duplicated(subset=["Action", "N° Dossier"], keep="first")
        to_remove_indices = df[not_zero_mask].loc[duplicated_mask].index
        removed_duplicates = df.loc[to_remove_indices]
        df = df.drop(index=to_remove_indices)
        nb_doublons = len(removed_duplicates)
        if nb_doublons > 0:
            st.warning(f"⚠️ {nb_doublons} doublon(s) supprimé(s).")
            st.subheader("🔍 Détails des doublons supprimés")
            st.dataframe(removed_duplicates.reset_index(drop=True))
        else:
            st.success("✅ Aucun doublon détecté.")
        df["Semaine"] = pd.to_numeric(df["Semaine"], errors="coerce").dropna().astype(int)
        data = df.groupby("Semaine").size().reset_index(name="Nombre")

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(data["Semaine"].astype(str), data["Nombre"], color=DARK_orange)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.5,
                str(int(height)), ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        ax.set_title("Dossiers clôturés par semaine", fontsize=14)
        ax.set_xlabel("Semaine")
        ax.set_ylabel("Nombre")
        plt.xticks(rotation=0)
        plt.tight_layout()

        display_chart_and_downloads(fig, data, "clotures_semaine.png", "clotures_semaine.csv", key_prefix="option4")
        if not data.empty:
            peak_week = data.loc[data["Nombre"].idxmax()]
            dashboard_insights.append(
                f"📈 [Clôtures élagage] : Pic de clôtures en semaine {peak_week['Semaine']} avec {peak_week['Nombre']} dossiers."
            )


    def plot_option5():
        df = pd.read_excel(xls, sheet_name="Traitement Dossier")
        df.columns = df.columns.str.strip()
        df["Heure de fin"] = pd.to_datetime(df["Heure de fin"], errors="coerce")
        df["Action"] = df["Action"].astype(str).str.strip().str.lower()
        df["N° Dossier"] = df["N° Dossier"].astype(str).str.strip()

        df = df[
            (df["Action"] == "relance 1  mail envoyé  à la maire") &
            (df["Heure de fin"] >= start_date) &
            (df["Heure de fin"] <= end_date)
        ]

        not_zero_mask = df["N° Dossier"] != "0"
        duplicated_mask = df[not_zero_mask].duplicated(subset=["Action", "N° Dossier"], keep="first")
        to_remove_indices = df[not_zero_mask].loc[duplicated_mask].index
        removed_duplicates = df.loc[to_remove_indices]
        df = df.drop(index=to_remove_indices)
        nb_doublons = len(removed_duplicates)
        if nb_doublons > 0:
            st.warning(f"⚠️ {nb_doublons} doublon(s) supprimé(s).")
            st.subheader("🔍 Détails des doublons supprimés")
            st.dataframe(removed_duplicates.reset_index(drop=True))
        else:
            st.success("✅ Aucun doublon détecté.")
        df["N°Semaine"] = pd.to_numeric(df["N°Semaine"], errors="coerce").dropna().astype(int)
        data = df.groupby("N°Semaine").size().reset_index(name="Nombre")

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(data["N°Semaine"].astype(str), data["Nombre"], color=LIGHT_BLUE)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.5,
                str(int(height)), ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        ax.set_title("Relance 1 à la mairie par semaine", fontsize=14)
        ax.set_xlabel("Semaine")
        ax.set_ylabel("Nombre")
        plt.xticks(rotation=0)
        plt.tight_layout()

        display_chart_and_downloads(
            fig,
            data.rename(columns={"N°Semaine": "Semaine"}),
            "relance_mairie.png",
            "relance_mairie.csv",
            key_prefix="option5"
        )
        if not data.empty:
            peak_week = data.loc[data["Nombre"].idxmax()]
            dashboard_insights.append(
                f"✉️ [Relances mairie] : Semaine {peak_week['N°Semaine']} a eu le plus de relances ({peak_week['Nombre']})."
            )



    if chart_option == "Flux entrant par semaine (Abandon doublon / Demande validé)":
        plot_option1()
    elif chart_option == "Répartition des 'en attente complément' par région":
        plot_option2()
    elif chart_option == "Taux de conformité par région":
        plot_option3()
    elif chart_option == "Dossiers clôturés par semaine (élagage réalisé)":
        plot_option4()
    elif chart_option == "Relance 1 Mail envoyé à la maire par semaine":
        plot_option5()
    elif chart_option == "Afficher tout":
        plot_option1()
        plot_option2()
        plot_option3()
        plot_option4()
        plot_option5()

# Ta fonction d'extraction du tableau
def extract_table_from_excel(file, sheet_name):
    # Lire la feuille sans en-tête
    df_raw = pd.read_excel(file, sheet_name=sheet_name, header=None)

    # Trouver l'index de la ligne où la 1ère colonne contient "Étiquettes de lignes"
    idx_header = df_raw[df_raw.iloc[:,0] == "Étiquettes de lignes"].index

    if idx_header.empty:
        st.error("La ligne 'Étiquettes de lignes' n'a pas été trouvée dans la première colonne.")
        return None

    header_row = idx_header[0]

    # La ligne header
    header = df_raw.iloc[header_row]

    # Le tableau commence juste après la ligne header
    df_table = df_raw.iloc[header_row+1:].copy()

    # Appliquer l'en-tête
    df_table.columns = header

    # Retirer les lignes vides (où toute la ligne est NaN)
    df_table.dropna(how='all', inplace=True)

    # Réindexer pour faciliter la lecture
    df_table.reset_index(drop=True, inplace=True)

    return df_table

st.title("Extraction tableau 'Étiquettes de lignes' & cumul par date")

# Upload du fichier Excel
uploaded_file = st.file_uploader("Chargez un fichier Excel", type=["xlsx"])

if uploaded_file:
    # Extraire la table
    df_table = extract_table_from_excel(uploaded_file, sheet_name="Relance de Jour")
    if df_table is not None:
        # Renommer la colonne de dates si nécessaire
        if "Étiquettes de lignes" in df_table.columns:
            df_table.rename(columns={"Étiquettes de lignes": "Date"}, inplace=True)

        st.subheader("Table extraite")
        st.dataframe(df_table)

        # Section cumul
        st.markdown("### 📅 Mouvement du Backlog")

        user_date_input = st.text_input(
            "Entrez une date au format **JJ/MM/AAAA** (ex: 07/07/2025)",
            placeholder="JJ/MM/AAAA"
        )

        if user_date_input:
            try:
                user_date = pd.to_datetime(user_date_input, format="%d/%m/%Y")
                df_table["Date"] = pd.to_datetime(df_table["Date"], format="%d/%m/%Y", errors="coerce")

                # Filtrer sur l'année sélectionnée
                df_year_filtered = df_table[df_table["Date"].dt.year == user_date.year].copy()
                df_year_filtered = df_year_filtered.sort_values("Date")

                cumulative_rows = []
                for _, row in df_year_filtered.iterrows():
                    date = row["Date"]
                    if pd.isna(date) or date > user_date:
                        break
                    try:
                        numeric_values = pd.to_numeric(row[1:], errors="raise")
                        cumulative_rows.append(numeric_values)
                    except:
                        break

                if cumulative_rows:
                    cumulative_df = pd.DataFrame(cumulative_rows)
                    summed = cumulative_df.sum()
                    summed_row = pd.DataFrame([summed])
                    summed_row.insert(0, "Date", f"Cumul jusqu'au {user_date.strftime('%d/%m/%Y')}")

                    st.success(f"✅ Cumul trouvé pour l'année {user_date.year} :")
                    st.dataframe(summed_row)

                    csv = summed_row.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📄 Télécharger le cumul CSV",
                        data=csv,
                        file_name=f"cumul_{user_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ Aucun cumul possible jusqu'à cette date (données manquantes ou invalides).")
            except ValueError:
                st.error("❌ Format de date invalide. Veuillez entrer une date comme **07/07/2025**.")
st.header("Génération automatique de rapport")

st.header("📋 Génération automatique du rapport")

if dashboard_insights:
    summary_text = summarize_for_llm(dashboard_insights)
    st.text_area("🧾 Résumé synthétique", summary_text, height=250)

    if st.button("🤖 Générer le rapport IA"):
        with st.spinner("⏳ Génération du rapport par Mistral..."):
            final_report = generate_llm_report(summary_text)
            st.subheader("📝 Rapport généré par IA")
            st.text_area("Rapport Manager", final_report, height=400)
            st.download_button(
                label="📄 Télécharger le rapport",
                data=final_report,
                file_name="rapport_manager.txt",
                mime="text/plain"
            )
else:
    st.info("Affichez d'abord les graphiques via 'Afficher tout' pour activer la génération du rapport.")
