import streamlit as st


import sys
import os

# Ajouter le dossier parent au path Python pour pouvoir importer utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import st_option as sto

st.set_page_config(page_title="Equity")

st.title("Equity")

# --- Entrée utilisateur pour le ticker ---
ticker = st.text_input("🎯 Entrez le ticker de l'action :", "AAPL").upper()
r = st.number_input("📉 Taux sans risque (%)", min_value=0.0, max_value=10.0, value=2.5) / 100  # converti en décimal

if ticker:
    # --- Partie générale ---
    st.header("🔍 Informations générales")
    col1, col2 = st.columns(2)

    with col1:
        nom = sto.get_name(ticker)
        if nom:
            st.markdown(f"**Nom de l’entreprise :** {nom} ({ticker})")
        prix = sto.get_last_price(ticker)
        if prix:
            st.metric(label="📌 Prix actuel", value=f"{prix:.2f} USD")
        var = sto.variation_depuis_ouverture(ticker)
        st.metric(label="📊 Variation depuis ouverture", value=f"{var} %", delta=f"{var} %")

    with col2:
        vol_actuel = sto.volume_depuis_ouverture(ticker)
        vol_moyen = sto.volume_moyen(ticker, "1mo")
        st.metric(label="🔄 Volume échangé aujourd’hui", value=f"{vol_actuel:,}")
        st.metric(label="📉 Volume moyen (1 mois)", value=f"{vol_moyen:,}")