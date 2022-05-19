from requests import options
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numerize import numerize


def app():
    st.markdown(""" # Marketing Campaign Analysis """)
    df = pd.read_csv("marketing_campaign.csv", delimiter=';')

    #====================================================================================#
    # preparing the data
    df['Dt_Customer']= pd.to_datetime(df['Dt_Customer'])

    def convert_col_type(dictionary,data):
        '''
        A function that converts the type of a given column/s in a dataframe, the input should be as a dictionary with columns as keys and the type as a value
        '''
        
        if len(dictionary)==1: # if there is one key-value pair, then there is no need to loop over the elements
            data[list(dictionary.keys())[0]]= data[list(dictionary.keys())[0]].astype(list(dictionary.values())[0]) # the left hand side is for accessing the required column = the right handside is changing its type
        else:
            for col, _type_ in dictionary.items():
                data[col]= data[col].astype(_type_)
        return None # I used return none to return nothing, because there is no need to return anything



    change_to = {'ID':'str'}
    convert_col_type(change_to,df)

    mean_income = df['Income'].mean()
    df.loc[df['Income'].isna(),'Income']=mean_income
    df.loc[df['Income']==df['Income'].max(),'Income']=mean_income

    # calculating frequency
    df['frequency'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
    # calculating Monetary Value
    df['Monetary'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    
    df['month_joined'] = df['Dt_Customer'].dt.month
    df['year_joined'] = df['Dt_Customer'].dt.year
    Months = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October' ,11:'November', 12:'December'}
    df['month_joined']=df['month_joined'].map(Months) 
    
    col_f1 ,col_f_income ,col_f2 = st.columns(3)
    with col_f1:
        with st.expander("Filter By Complaints"):
            complains_filter = st.radio(label='Choose a group',
            options=("All","Filter by those who complained","Filter by those who did not complain"))
    with col_f_income:
        with st.expander("Filter By Income"):
            values_income_level = st.slider('Select a range of values', 0.0 , 200000.0, (0.0, 200000.0))
    
    with col_f2:    
        with st.expander("Filter By Respondents"):
            responds_filter = st.radio(label="Choose a group",options=
            ("All","Filter by those who responded","Filter by those who did not respond"))
    

    if complains_filter=="Filter by those who did not complain" :
        df=df[df['Complain']==0]
    if complains_filter=="Filter by those who complained" :
        df=df[df['Complain']==1]
    if responds_filter == "Filter by those who did not respond":
        df=df[df['Response']==0]
    if responds_filter == "Filter by those who responded":
        df=df[df['Response']==1]
    
    df=df[(df['Income']>=values_income_level[0]) & (df['Income']<=values_income_level[1])] 
    
    total_per_category=(df[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']].sum()).to_frame()
    total_per_category.reset_index(inplace=True)
    total_per_category.rename(columns={'index':'Category',0:'Total Revenue'},inplace=True)

    st.write(" ")
    campaigns_summary = (df[['AcceptedCmp1','AcceptedCmp2' ,'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum().to_frame())
    campaigns_summary.reset_index(inplace=True)
    campaigns_summary.rename(columns={'index':'Campaign', 0:'Total Accepted'},inplace=True)
    


    col_m1 , col_m2 , col_m3 , col_m4 = st.columns(4)
    theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}
    with col_m1:
        st.metric(label="Total Number Of Clients", value= numerize.numerize(float(df.shape[0])))
        #hc.info_card(title='Total Number Of Clients', content=df.shape[0],theme_override=theme_good)
        st.metric(label="Total Number Of Complaints", value = df['Complain'].sum())
    
    with col_m2:
        st.metric(label="Average Recency" , value = round(df['Recency'].mean(),2))
        #hc.info_card(title='Average Recency', content=round(df['Recency'].mean(),2),theme_override=theme_good)
        
        st.metric(label="Total Number Of Responses", value = df['Response'].sum())
        #hc.info_card(title='Total Number Of Responses', content=df['Response'].sum(),theme_override=theme_good)    
    with col_m3:
        
        st.metric(label="Average Frequency" , value = round(df['frequency'].mean(),2))
        #hc.info_card(title='Average Frequency', content=round(df['frequency'].mean(),2),theme_override=theme_good)

        st.metric(label="Total Revenue", value = numerize.numerize(float(total_per_category['Total Revenue'].sum())))
        #hc.info_card(title='Total Revenue', content=total_per_category['Total Revenue'].sum(),theme_override=theme_good)

    with col_m4:
        st.metric(label="Average Monetary" , value = round(df['Monetary'].mean(),2))
        #hc.info_card(title='Average Monetary', content=round(df['Monetary'].mean(),2),theme_override=theme_good)
        st.metric(label= "Max Responses From Old Campaigns", value = campaigns_summary['Total Accepted'].max())

    
    
    st.markdown(
        """
    <style>
    div.css-1xarl3l.e16fv1kl2{
            
            color: #636efa;
            font-size: 30px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
    div#stHorizontalBlock.css-ocqkz7.e1tzin5v0 {
            border-bottom-width: 2px;
            border-bottom-style: solid;
            border-bottom-color: #1C6EA4;
            
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    

    st.markdown(
        """
    <style>
    div#stMetricLabel.css-1rh8hwn.e16fv1kl1{
            
            color: #636efa;
            font-size: 60px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    





    


    
    month_counts = (df['month_joined'].value_counts()).to_frame()
    month_counts.reset_index(inplace=True)
    month_counts.rename(columns={'index':'Months' , 'month_joined':'Total New Clients'},inplace=True)

    col_bar_month ,col_void ,col_bar_category = st.columns([5,0.5,5])
    with col_bar_month:
        fig_month_counts = px.bar(month_counts, x="Months", y='Total New Clients',text_auto=True)
        fig_month_counts.update_layout(
                    title_text = "New Clients Per Month")
    

        st.plotly_chart(fig_month_counts,use_container_width=True)

        with st.expander(label= 'Interpretation'):
            st.write(list((month_counts.sort_values(by='Total New Clients',ascending=False))['Months'])[0],
            "is the peak month in which clients join our business, whereas,",list((month_counts.sort_values(by='Total New Clients',ascending=False))['Months'])[-1],
            "is the month with the lowest number of new clients,")

    with col_bar_category:
        # most profitable category 
        fig_category_counts = px.bar(total_per_category.sort_values(by='Total Revenue',ascending=False), x="Category", y='Total Revenue',text_auto=True)
        fig_category_counts.update_layout(
                    title_text = "Total Revenue Per Category")
        st.plotly_chart(fig_category_counts,use_container_width=True)
        with st.expander(label='Interpretation'):
            st.write(list((total_per_category.sort_values(by='Total Revenue',ascending=False))['Category'])[0],
            "is the category with the most revenue, whereas.",
             list((total_per_category.sort_values(by='Total Revenue',ascending=False))['Category'])[-1],
             "is the category with the lowest generated revenue.")      

    st.write(" ")

    
    campaigns_summary = (df[['AcceptedCmp1','AcceptedCmp2' ,'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum().to_frame())
    campaigns_summary.reset_index(inplace=True)
    campaigns_summary.rename(columns={'index':'Campaign', 0:'Total Accepted'},inplace=True)
    total_accepted_5=np.sum(list(campaigns_summary['Total Accepted']))

    initial = list(campaigns_summary[campaigns_summary['Campaign']=='AcceptedCmp1']['Total Accepted'].values)
    row_diff=campaigns_summary['Total Accepted'].diff()[1:].to_list()
    for item in row_diff:
        initial.append(item)
    initial.append(total_accepted_5)

    list_test = [str(item) for item in initial[:-1]]
    list_test.append(str(sum(initial[:-1])))
    campaign_number=campaigns_summary['Campaign'].to_list()
    campaign_number.append("Total")
    

    fig2  = go.Figure()
    hrz = campaign_number

    vrt  = initial
    text = list_test
    fig2.add_trace(go.Waterfall(
                x = hrz, y = vrt,
                base = 0,
                text = text, textposition = 'inside',   
                measure = ["absolute",  "relative", "relative","relative", "relative" ,"total"]  
                ))              
    fig2.update_layout(
                    title_text = "Number of acceptances",
                    # title_font=dict(size=25,family='Verdana', 
                    #                 color='darkred')
                    )
    st.plotly_chart(fig2,use_container_width=True)


    
