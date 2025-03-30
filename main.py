import streamlit as st
from modules.refiner import refiner

# Set up the sidebar
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ["Home", "Refiner", "Settings", "About"],
    index=1
)

# Main content based on selection
if selected_page == "Home":
    st.title("Welcome to the Application")
    st.write("Select a page from the sidebar to get started.")
elif selected_page == "Refiner":
    refiner.run()
elif selected_page == "Settings":
    st.title("Settings")
    st.write("Settings page coming soon...")
else:
    st.title("About")
    st.write("About page coming soon...")
