import streamlit as st
import pandas as pd
import numpy as np

st.title("🏥 Healthcare Analytics - TEST")
st.write("If you can see this, the app is working!")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Basic Stats")
    st.dataframe(df.describe())
