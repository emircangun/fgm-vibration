import streamlit as st
from model import *

@st.cache_resource
def gui_load_model(model_path):
    model = load_model(model_path)
    return model

