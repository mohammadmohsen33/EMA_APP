import streamlit as st
import streamlit.components.v1 as components
from pytrends.request import TrendReq
import plotly.express as px

# embed streamlit docs in a streamlit app
def app():
    st.title("Competitor And Trend Analysis")
    col_tz ,col_void_t ,col_kw1 = st.columns([5,0.5,5])
    with col_tz:
        time_zone=st.text_input(label="Enter your timezone")
    with col_kw1:
        word = str(st.text_input("Enter a keyword"))
    
    with st.expander(label="Compare to another keyword"):
        word2 = str(st.text_input("Enter a keyword",key="second keyword"))
    kw_list = []     
    if ((len(time_zone) > 0) and (len(word) > 0) and (len(word2) == 0)): 
        kw_list.append(word)
        pytrends = TrendReq(hl='en-US', tz=time_zone)
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
        #1 Interest over Time
        data = pytrends.interest_over_time() 
        data = data.reset_index()
        fig = px.line(data, x="date", y=kw_list, title='Keyword Web Search Interest Over Time')
        st.plotly_chart(fig,use_container_width=True) 
    elif ((len(time_zone) > 0) and (len(word) > 0) and (len(word2) > 0)) :
        kw_list.append(word)
        kw_list.append(word2)
        pytrends = TrendReq(hl='en-US', tz=time_zone)
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
        data = pytrends.interest_over_time() 
        data = data.reset_index()
        fig = px.line(data, x="date", y=kw_list, title='Keyword Web Search Interest Over Time')
        st.plotly_chart(fig,use_container_width=True)
       
    else:
        st.write("Please enter valid values")
        
    
     
   

         
