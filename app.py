import streamlit as st
from multiapp import MultiApp
from apps import home_page,data_table, home,descriptive ,Customer_segmentation, Google_Trends # import your app modules here
import pandas as pd 
import numpy as np 
import plotly.express as px

app = MultiApp()

# Add all your application here
app.add_app("Intro Page", home_page.app)
app.add_app("Upload Data", data_table.app)
app.add_app("Dashboard", home.app)
app.add_app("Exploration & Modelling", descriptive.app)
app.add_app("Customer Segmentation", Customer_segmentation.app)
app.add_app("Competitor & Trend analysis", Google_Trends.app)

st.markdown('''<style>.css-1sls4c0 {border-top: 60px solid #0d6efd;}<style> ''',unsafe_allow_html=True)
st.markdown('''<style>.css-1oglvfc.e16fv1kl1 {font-size: 17px;}<style> ''',unsafe_allow_html=True)
st.markdown('''<style>.js-plotly-plot .plotly .user-select-none {border-top: 2px solid #0d6efd;}<style> ''',unsafe_allow_html=True)
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 0rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;     
                }
        </style>
        """, unsafe_allow_html=True)
#app.add_app("Model", model.app)
#app.add_app("Model", model.app)
#app.add_app("Model", model.app)
# The main app
app.run()
