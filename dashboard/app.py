import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="CrisisWatch Indonesia", page_icon="🚨")

st.markdown("""
<style>
.badge-flood { background-color: #3B82F6; color: white; padding: 2px 6px; border-radius: 4px; }
.badge-earthquake { background-color: #EF4444; color: white; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("🚨 CrisisWatch Indonesia Dashboard")
st.sidebar.title("Filters")
st.sidebar.markdown("Use the navigation above to explore different views.")

st.markdown("### Welcome to CrisisWatch")
st.write("This dashboard monitors social media and news feeds for crisis events in Indonesia.")
