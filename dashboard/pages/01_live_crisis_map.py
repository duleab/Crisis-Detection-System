import streamlit as st
import folium
from streamlit_folium import st_folium

st.header("🗺️ Live Crisis Map")

m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="CartoDB positron")
folium.Marker([-6.2, 106.8], popup="Jakarta Flood").add_to(m)

st_data = st_folium(m, width="100%", height=600)
