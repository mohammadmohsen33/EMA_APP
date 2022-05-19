import pandas as pd 
import numpy as np 
import streamlit as st 
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


def app():
    st.title("Interactive Data Table")
    col1_upload, col2_void, col3_void = st.columns(3)
    
    with col1_upload:
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    if uploaded_files == None:
        st.write("Please Upload a csv file")
    else:
        data_tab = pd.read_csv(uploaded_files,delimiter=';')
        
        data_tab['Dt_Customer']= pd.to_datetime(data_tab['Dt_Customer'])

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
        convert_col_type(change_to,data_tab)

        mean_income = round(data_tab['Income'].mean(),2)
        data_tab.loc[data_tab['Income'].isna(),'Income']=mean_income
        data_tab.loc[data_tab['Income']==data_tab['Income'].max(),'Income']=mean_income

        # calculating frequency
        data_tab['frequency'] = data_tab['NumDealsPurchases'] + data_tab['NumWebPurchases'] + data_tab['NumCatalogPurchases'] + data_tab['NumStorePurchases']
        # calculating Monetary Value
        data_tab['Monetary'] = data_tab['MntWines'] + data_tab['MntFruits'] + data_tab['MntMeatProducts'] + data_tab['MntFishProducts'] + data_tab['MntSweetProducts'] + data_tab['MntGoldProds']
        
        
        gb = GridOptionsBuilder.from_dataframe(data_tab)
        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
        gb.configure_side_bar() #Add a sidebar
        gridOptions = gb.build()

        grid_response = AgGrid(
            data_tab,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT', 
            update_mode='MODEL_CHANGED', 
            fit_columns_on_grid_load=False,
            theme='blue', #Add theme color to the table
            enable_enterprise_modules=True,
            height=350, 
            reload_data=True
        )

        data_tab = grid_response['data']


