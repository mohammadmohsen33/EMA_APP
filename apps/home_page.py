from PIL import Image
import streamlit as st
import base64
import json
import requests
from streamlit_lottie import st_lottie
import webbrowser


def app():
    # giving tha page a title
    st.title("Explore, Model And Analyze. All In One Tool.")

    st.write(" ")
    # defining a function that loads a lottiefile
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    
    # loading the lottiefile from directory
    lottie_file = load_lottiefile("business-analysis.json")

    # creating columns to modify the layout of the app
    col_text ,col_empty ,col_lottie = st.columns([5,0.5,5])
    
    with col_text:
        st.write("In this app, the user can explore the dataset after uploading it.\
            and use the power of machine learning to segment entites and label unlabeled\
                 datasets and download them.\
                      The user could also do some predictive analytics to predict a\
                certain variable. The app also offer a quick access to Google\
                    trends for easy competitor and trend analysis.")
        
        st.write(" ")

        st.write("By Mohammad Mohsen")
        contact_me=st.button(label="Get in touch")

        if contact_me:
            webbrowser.open_new_tab("mailto:mim26@mail.aub.edu")


    
    with col_lottie:
        st_lottie(lottie_file)
    