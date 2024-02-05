import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Trading app",
    page_icon="👋",
)

st.write("# Welcome to Trading App! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Trading App is an open-source app built specifically for
    Machine Learning and Data Science with time-series projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Tradin App can do!
    ### Want to learn more?
    - Explore more project [gitlab.com/DalvikLoger](https://www.gitlab.com/DalvikLoger)
""")

