from ast import With
import pandas as pd 
import numpy as np 
import streamlit as st 
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import base64 
import time



def app():
  # loading the data
  timestr = time.strftime("%Y%m%d-%H%M%S")
  st.title("Customer Segmentation")
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

  
  def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


  
    



  change_to = {'ID':'str'}
  convert_col_type(change_to,df)

  mean_income = df['Income'].mean()
  df.loc[df['Income'].isna(),'Income']=mean_income
  df.loc[df['Income']==df['Income'].max(),'Income']=mean_income

  # calculating frequency
  df['frequency'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
  # calculating Monetary Value
  df['Monetary'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']



  #====================================================================================#
  df_copy_clustering = df.copy()
  df_copy_clustering.drop(columns=['ID','Education','Dt_Customer',
        'Marital_Status','Z_Revenue','Z_CostContact','Response','Complain','Year_Birth','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2'],inplace=True)

  st.write("In this page, the user can utilize the power of machine learning to cluster the customers.\
            The user can choose any features depending on the business use case, and can specify the number of clusters.")

  col_features , col_void ,col_slider = st.columns([5,0.5,5])
  with col_features:
    features_to_cluster = st.multiselect(label="Select features to segment the clients",options= df_copy_clustering.columns)
  with col_slider:
    segment_number =st.slider(label="Select the number of segments", min_value=2, max_value=10, step=1 )
  
  
  col_clusters , col_void ,col_interpretation = st.columns([5,0.5,5])
  
  if len(features_to_cluster)>1:

    with col_clusters:
      X = df[features_to_cluster]
      estimator = KMeans(n_clusters=segment_number)
      estimator.fit(X)
      labels = estimator.labels_
      if (len(features_to_cluster)==2) :
        sil_1=round(silhouette_score(X,labels),2)
        fig_segmentation = px.scatter(X, x=features_to_cluster[0], y=features_to_cluster[1],
                      color=labels.astype(str))
        fig_segmentation.update_layout(title_text=f'The silhouette score is:{sil_1}')
        st.plotly_chart(fig_segmentation,use_container_width=True)


      elif (len(features_to_cluster)==3):
        sil_2=round(silhouette_score(X,labels),2)
        fig_segmentation = px.scatter_3d(X, x=features_to_cluster[0], y=features_to_cluster[1], z=features_to_cluster[2],
                      color=labels.astype(str))
        fig_segmentation.update_layout(title_text=f'The silhouette score is:{sil_2}')
        st.plotly_chart(fig_segmentation,use_container_width=True)
      
      else:
        st.write("The number of features cannot be visualized")
        sil_3=round(silhouette_score(X,labels),2)
        st.write(f"The silhouette score is:{sil_3}")
      
    
    with col_interpretation:
      with st.expander("Interpretation"):
        st.write("""
        Using machine learning, we were able to cluster our clients into""",segment_number,''' clusters ''' )
        
      clustering_dataframe = pd.DataFrame(estimator.cluster_centers_ , columns = features_to_cluster)
      st.write("Here are some information regrading the resulting clusters, with numbers representing the avergae score for clients within the correspondent cluster")
      st.dataframe(clustering_dataframe)
    

        

  else:
    st.write("Please select variables to cluster the clients")

  col_dow_void_1 ,col_download , col_dow_void_2 = st.columns([1,1,3])

  
  with col_dow_void_2:
    label_the_data = st.checkbox("Label the data to download",value=False)

    if label_the_data:
    
      df_segmented = df.copy()
      df_segmented['Customer Segment'] = estimator.labels_
      
      csv = convert_df(df_segmented)

      st.download_button(
          label="Download data as CSV",
          data=csv,
          file_name='Labeled_data.csv',
          mime='text/csv',
      )

  

 
