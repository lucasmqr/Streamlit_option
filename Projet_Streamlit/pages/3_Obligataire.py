import streamlit as st

import sys
import os

# Ajouter le dossier parent au path Python pour pouvoir importer utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import st_option as sto

import sys
import os

# Ajouter le dossier parent au path Python pour pouvoir importer utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="Obligataire")

st.title("Obligataire")

# from datetime import date

# start_year = date.today().replace(month=1, day=2)
# today = date.today()

# tickers_dict = {
#     "USA": {
#         "3M": "^IRX",
#         "5Y": "^FVX",
#         "10Y": "^TNX",
#         "30Y": "^TYX"
#     },
#     "France": {
#         "2Y": "FR2YT=RR",
#         "5Y": "FR5YT=RR",
#         "10Y": "FR10YT=RR",
#         "30Y": "FR30YT=RR"
#     },
#     "Germany": {
#         "2Y": "DE2YT=RR",
#         "5Y": "DE5YT=RR",
#         "10Y": "DE10YT=RR",
#         "30Y": "DE30YT=RR"
#     }
# }

# if st.button("📊 Afficher la courbe des taux"):
#     data = sto.get_yield_curve(tickers_dict, start_year, today)
#     sto.plot_yield_curves(data)
