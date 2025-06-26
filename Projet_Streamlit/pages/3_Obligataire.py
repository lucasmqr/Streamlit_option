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