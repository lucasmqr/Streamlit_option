import streamlit as st
import st_option as sto

st.set_page_config(page_title="Analyse Financière", layout="wide")

st.title("📈 Analyse Financière Complète")

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

    # --- Partie Option ---
    st.header("📘 Options & Volatilité Implicite")

    with st.expander("📈 Visualiser les options disponibles"):
        df_opt, msg = sto.get_options_dataframe(ticker)
        if not df_opt.empty:
            st.write(df_opt.head())
        else:
            st.warning(msg)

    with st.expander("🧮 Calculer le prix d’une option (Black-Scholes)"):
        K = st.number_input("Strike (K)", value=prix if prix else 100.0)
        T = st.number_input("Maturité (en années)", value=0.5)
        sigma = st.number_input("Volatilité implicite estimée (σ)", value=0.2)
        opt_type = st.selectbox("Type d’option", ['call', 'put'])
        if st.button("Calculer le prix BS"):
            price = sto.black_scholes(ticker, K, T, r, sigma, opt_type)
            st.success(f"💰 Prix théorique ({opt_type}) : {price} USD")

    with st.expander("🔎 Volatilité implicite d'une option"):
        market_price = st.number_input("Prix observé du marché", value=10.0)
        if st.button("Calculer la volatilité implicite"):
            iv = sto.get_implied_volatility(ticker, K, T, r, market_price, opt_type)
            if iv:
                st.success(f"Volatilité implicite estimée : {iv:.4f}")
            else:
                st.error("⚠️ Échec de convergence.")

    # --- Partie Graphique (Nappe, Smile, Skew) ---
    st.header("🌐 Visualisation de la volatilité")

    if not df_opt.empty:
        col3, col4 = st.columns(2)

        with col3:
            if st.button("Afficher la nappe de volatilité (surface 3D)"):
                sto.plot_vol_surface(df_opt, ticker, r, option_type=opt_type)

        with col4:
            if st.button("Afficher le smile et le skew"):
                sto.plot_smile_and_skew(df_opt, ticker, r, option_type=opt_type)

# 🚀 Ajouts futurs : partie obligations, modèle de Heston, CDS, etc.
