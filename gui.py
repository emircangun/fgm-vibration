import streamlit as st

from utils import *
from config import MODEL_WEIGHTS_FILE
import config
from train_and_evaluate import *


st.set_page_config(layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Lambda Prediction</h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    methods = st.radio(
        "Method",
        [method for method in config.METHOD_DICT.keys()]
    )

with col2:
    beam_types = st.radio(
        "Beam Type",
        [beam_type for beam_type in config.BEAM_TYPE_DICT.keys()]
    )

with col3:
    l_h = st.radio(
        "L/h Ratio",
        [l_h for l_h in config.L_H_DICT.keys()]
    )

with col4:
    k_value = st.number_input("k value", value=0.0)
    

clicked = st.button('Predict Lambda')

# prediction logic
model = gui_load_model(MODEL_WEIGHTS_FILE)
if clicked:
    inp = [methods, beam_types, l_h, k_value]
    out = test_examples(model, [inp])[0]

    # st.write(f"Predicted lambda: {out}")
    st.markdown(f"<h3 style='text-align: center; color: white;'>Predicted lambda:  {out.item()}</h3>", unsafe_allow_html=True)
