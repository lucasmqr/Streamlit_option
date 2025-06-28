import streamlit as st
import sys
import os

# Import module depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import st_option as sto

st.set_page_config(page_title="Option")
st.title("Options")

# Entrées utilisateur
ticker = st.text_input("Entrez le ticker de l'action :", "AAPL").upper()
r = st.number_input("Taux sans risque (%)", min_value=0.0, max_value=10.0, value=2.5) / 100

st.markdown("---")

col1,col2 =st.columns(2)


with col1:
    st.write("📘 Calcul du prix d'une option (Black-Scholes)")

    sp = sto.get_last_price(ticker)[0]
    strike_bs = st.number_input("Strike (BS)", value=int(sp + 10))
    maturity_bs = st.number_input("Maturité (en années) (BS)", value=1.0)
    vol_bs = st.number_input("Volatilité en % (BS)", value=20.0) / 100
    type_bs = st.selectbox("Type d’option (BS)", ["call", "put"])
    spot_bs = st.number_input("Spot (BS)", value=float(sp))

    if st.button("Calculer le prix avec Black-Scholes"):
        prix_bs = sto.black_scholes(spot_bs, None, strike_bs, maturity_bs, r, vol_bs, type_bs)
        if prix_bs is not None:
            st.metric(label="🎯 Prix théorique BS", value=f"{prix_bs:.2f} USD")
        else:
            st.error("Erreur dans le calcul. Vérifiez les paramètres.")
    
    greeks = sto.black_scholes_greeks(spot_bs, strike_bs, maturity_bs, r, vol_bs, type_bs)
    if greeks:
        st.subheader("Grecques")

        st.metric("Delta", f"{greeks['Delta']:.4f}")
        st.metric("Vega", f"{greeks['Vega']:.4f}")
        st.metric("Rho", f"{greeks['Rho']:.4f}")
        st.metric("Theta (par jour)", f"{greeks['Theta']:.4f}")


with col2:
    st.write("📈 Calcul de la volatilité implicite")

    strike_iv = st.number_input("Strike (VI)", value=int(sp + 10))
    maturity_iv = st.number_input("Maturité (en années) (VI)", value=1.0)
    market_price = st.number_input("Prix observé sur le marché (VI)")
    type_iv = st.selectbox("Type d’option (VI)", ["call", "put"])
    spot_iv = st.number_input("Spot (VI)", value=float(sp))

    if st.button("Calculer la volatilité implicite"):
        vi = sto.get_implied_volatility(spot_iv, strike_iv, maturity_iv, r, market_price, type_iv)
        if vi:
            st.metric(label="🔎 Volatilité implicite", value=f"{vi:.2%}")
        else:
            st.error("⚠️ Erreur de convergence pour la volatilité implicite.")












# Charger les données d'options
df_opts = sto.get_all_options(ticker)



#     prix = sto.get_last_price(ticker)[0]

#     st.header("Options & Volatilité Implicite")

#     with st.expander("📈 Visualiser les options disponibles"):
#         df_opt, msg = sto.get_options_dataframe(ticker)
#         if not df_opt.empty:
#             st.write(df_opt.head())
#         else:
#             st.warning(msg)

#     with st.expander("🧮 Calculer le prix d’une option (Black-Scholes)"):
#         K = st.number_input("Strike (K)", value=prix if prix else 100.0)
#         T = st.number_input("Maturité (en années)", value=0.5)
#         sigma = st.number_input("Volatilité implicite estimée (σ)", value=0.2)
#         opt_type = st.selectbox("Type d’option", ['call', 'put'])
#         if st.button("Calculer le prix BS"):
#             price = sto.black_scholes(ticker, K, T, r, sigma, opt_type)
#             st.success(f"💰 Prix théorique ({opt_type}) : {price:.2f} USD")

#     with st.expander("🔎 Volatilité implicite d'une option"):
#         market_price = st.number_input("Prix observé du marché", value=10.0)
#         if st.button("Calculer la volatilité implicite"):
#             iv = sto.get_implied_volatility(ticker, K, T, r, market_price, opt_type)
#             if iv:
#                 st.success(f"Volatilité implicite estimée : {iv:.4f}")
#             else:
#                 st.error("⚠️ Échec de convergence.")

#     st.header("🌐 Visualisation de la volatilité")

#     if not df_opt.empty:
#         col3, col4 = st.columns(2)

#         with col3:
#             if st.button("Afficher la nappe de volatilité (surface 3D)"):
#                 sto.plot_vol_surface(df_opt, ticker, r, option_type=opt_type)

#         with col4:
#             if st.button("Afficher le smile et le skew"):
#                 sto.plot_smile_and_skew(df_opt, ticker, r, option_type=opt_type)
