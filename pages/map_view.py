import streamlit as st

def store_value(key):
    st.session_state[key] = st.session_state["_"+key]
def load_value(key):
    st.session_state["_"+key] = st.session_state[key]

distance = load_value("distance")
custom_distance = load_value("custom_distance")

st.write(type(distance),distance)
st.write(type(custom_distance),custom_distance)

st.sidebar.markdown("# Map View")

