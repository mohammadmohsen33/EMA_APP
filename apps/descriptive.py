import pandas as pd 
import numpy as np 
import streamlit as st 
import plotly.express as px
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from streamlit_option_menu import option_menu


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc



def app():
    # a horizantal menu to switch between the exploration and response prediction pages
    selected2 = option_menu(None, ["Exploration", "Response Prediction"], 
    icons=['bar-chart',"lightbulb"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

    
    
    
    data_tab = pd.read_csv('Marketing_campaign.csv',delimiter=';')
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

    df_copy = data_tab.copy()
    df_copy.drop(columns=['ID','Education','Dt_Customer',
        'Marital_Status','Z_Revenue','Z_CostContact','Response','Complain','Year_Birth','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2'],inplace=True)

    
    
    if selected2 == 'Exploration':
        st.title("Explore The Dataset")
        st.write("In this page, the user can have the full control to explore the dataset and the relationship between variables.")
        pio.templates.default = "plotly_white"

        col_num , col_void, col_cat = st.columns([5,0.5,5])

        with col_num:
            options = st.multiselect("Choose numerical variables to explore",options=list(df_copy.columns))   
            if len(options)==1:
                type_of_chart = st.radio(label="Type of chart" , options=('Boxplot','Histogram'))
                if type_of_chart == 'Boxplot':
                    fig_num_1 = px.box(df_copy,y= options[0])
                    fig_num_1.update_layout(
                        title_text = f"Boxplot visualizing the distribution of {options[0]}")
                    st.plotly_chart(fig_num_1,use_container_width=True)
            
                if type_of_chart == 'Histogram':
                    fig_num_2 = px.histogram(df_copy,x= options[0])
                    fig_num_2.update_layout(
                        title_text = f"Histogram visualizing the distribution of {options[0]}")
                    st.plotly_chart(fig_num_2,use_container_width=True)   
            
            elif len(options)==2:
                fig_num_3 = px.scatter(df_copy[options], x=options[0] , y=options[1])
                fig_num_3.update_layout(
                        title_text = f"Scatter plot visualizing {options[0]} vs {options[1]}")
                fig_num_3.update_layout()
                st.plotly_chart(fig_num_3,use_container_width=True)

            elif len(options)>2:
                df_corr = df_copy[options].corr()

                
                x = list(df_corr.columns)
                y = list(df_corr.index)
                z = np.array(df_corr)

                fig_num_4 = ff.create_annotated_heatmap(
                    z,
                    x = x,
                    y = y ,
                    annotation_text = np.around(z, decimals=2),
                    colorscale='Brwnyl',
                    showscale=True
                    )
                st.plotly_chart(fig_num_4,use_container_width=True)
            else:
                st.write(" ")


        df_copy_2 =data_tab[['Education','Marital_Status','Response','Complain','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2']].copy()

        
        
        convert_col_type({'Education':'category','Marital_Status':'category',
        'Response':'str','Complain':'str','AcceptedCmp3':'str',
            'AcceptedCmp4':'str', 'AcceptedCmp5':'str', 'AcceptedCmp1':'str',
            'AcceptedCmp2':'str'},df_copy_2)
        
        with col_cat:
            options_2 = st.multiselect("Choose categorical variables to explore",options=list(df_copy_2.columns))

            if len(options_2)== 1:
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                fig_cat_1 = px.histogram(data_frame=df_copy_2 , x= options_2[0])
                fig_cat_1.update_layout(
                        title_text = f"Count plot for {options_2[0]}")
                st.plotly_chart(fig_cat_1,use_container_width=True)
            
            elif len(options_2) == 2:
                df_cross = pd.crosstab(df_copy_2[options_2[0]] , df_copy_2[options_2[1]],normalize='index')
                # initiate data list for figure
                data = []
                #use for loop on every zoo name to create bar data
                for x in df_cross.columns:
                    data.append(go.Bar(name=str(x), x= df_cross.index, y=df_cross[x]))

                figure = go.Figure(data)
                figure.update_layout(barmode = 'group')
                figure.update_layout(
                        title_text = f"Clustered bar chart for {options_2[0]} and {options_2[1]}")
                figure.update_layout(
                    dict(xaxis_title=options_2[0])
                )
                figure.update_layout(dict(legend_title_text=options_2[1]))

                #For you to take a look at the result use
                st.plotly_chart(figure,use_container_width=True)
            
            elif len(options_2) > 2 :
                st.warning("You have to select at most 2 variables")

            else:
                st.write(" ")
        

        # here is a section to explore the relationship between numerical and categorical variables
        
        if ((len(options)== 1) & (len(options_2)== 1)):
            fig = px.box(data_tab, x=options_2[0], y= options[0])
            fig.update_layout(
                        title_text = f"Side by side Boxplot for {options_2[0]} and {options[0]}")
            st.plotly_chart(fig,use_container_width=True)
            
        else:
            st.warning("Explore the relation between a categorical and a numerical variables. Choose one for each type.")

    

    if selected2 == 'Response Prediction':
        # instantiating the model
        pred_model = RandomForestClassifier(n_estimators=1000)
        # a vector containing the target variable
        target_dataset = data_tab['Response']

        st.title("Predict who will respond")

        st.write("In this page, the user can utilize the power of machine learning to predict whether a customer will respond.\
            The user can choose any features depending on the availability of the customer's data.")

        df_copy_2 =data_tab[['Education','Marital_Status','Complain','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2']].copy()
        
        col_num_features , col_void_pred, col_cat_features = st.columns([5,0.5,5])
        
        with col_num_features:
            Num_features = st.multiselect("Choose numerical features to train the machine learning model",options=list(df_copy.columns))


        with col_cat_features:
            cat_features = st.multiselect("Choose categorical features to train the machine learning model",options=list(df_copy_2.columns))
        
        
        col_train ,col_vide ,col_new_instance = st.columns([5,0.5,5])

        if len(Num_features+cat_features) > 0:

            all_features = Num_features + cat_features
            
            training_data = pd.get_dummies(data_tab[all_features])

            with st.expander("Preview preprocessed feature dataset"):
                st.dataframe(training_data)
            
            

            with col_train:

                Train_now =st.checkbox(label= "Train the model on the selected features")

                if Train_now:
                    scores = cross_val_score(pred_model,training_data, target_dataset ,cv=5 ,scoring='roc_auc', n_jobs=-1) 
                    # reporting average auc
                    st.write(f"The model is evaluated\
                        using cross validation and achieved an AUC: {np.mean(scores)*100:.3f} ± {np.std(scores)*100:.3f}" )
                    
                    X_train, X_test , y_train, y_test = train_test_split(training_data , target_dataset , train_size=0.75 , shuffle = True , stratify = target_dataset)

                    pred_model.fit(X_train , y_train)
                    # Now we will allow the user to see the Roc curve, for him to choose the best threshhold
                    y_score = pred_model.predict_proba(X_test)[:, 1]

                    fpr, tpr, thresholds = roc_curve(y_test, y_score)
                    conf_data=pd.DataFrame({"False Positive Rate":list(fpr), "Thresholds":list(thresholds),"True Positive Rate":list(tpr)})
                    

                    fig = px.area(data_frame=conf_data,
                        x="False Positive Rate", y="True Positive Rate",
                        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',hover_data = ['Thresholds'],
                        labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        width=700, height=500
                    )
                    fig.add_shape(
                        type='line', line=dict(dash='dash'),
                        x0=0, x1=1, y0=0, y1=1
                    )

                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    fig.update_xaxes(constrain='domain')
                    
                    st.plotly_chart(fig,use_container_width=True)



                



        

        # else:
        #     st.write(" ")
            with col_new_instance:

                train_new_insance = st.checkbox("Classify new client")
                st.write(" ")

                if train_new_insance:
                        #preparing the new data for the user to input any user information for classification.
                
                    if not Train_now:            
                        selected_threshhold = st.slider(label="Choose a suitable threshold", min_value=0.0 , max_value=1.0, step=0.01,value= 0.5)

                        list_of_numeric_inputs = []
                        for numerical_chosen in Num_features:
                            numerical_feat_input="numerical_feat_input"+str(Num_features.index(numerical_chosen))
                            numerical_feat_input = st.number_input(f'Insert the client {numerical_chosen}')
                            list_of_numeric_inputs.append(numerical_feat_input)
                        
                        list_of_cat_inputs = []
                        for catergorical_chosen in cat_features:
                            cat_feat_input = "cat_feat_input" +str(cat_features.index(catergorical_chosen))
                            cat_feat_input = st.selectbox(label=f'Insert the client {catergorical_chosen}', options= tuple(data_tab[catergorical_chosen].unique()))
                            list_of_cat_inputs.append(cat_feat_input)

                        new_instance=[]
                        for num_input in list_of_numeric_inputs:
                            new_instance.append(num_input)
                        for cat_input in list_of_cat_inputs:
                            new_instance.append(cat_input)
                        st.write(new_instance)

                        new_ins_dataframe= data_tab[all_features].copy()
                        
                        df_length = len(new_ins_dataframe)
                        new_ins_dataframe.loc[df_length] = new_instance

                        
                        test_ins=pd.get_dummies(new_ins_dataframe)
                        client_to_classify = test_ins[test_ins.index==(len(test_ins)-1)]
                        # training the model on the whole dataset, after the user has approved its performance on the cross validation
                        pred_model.fit(training_data,target_dataset)
                        #st.write(pred_model.predict_proba(client_to_classify))
                        

                        y_pred = (pred_model.predict_proba(client_to_classify)[:, 1] > selected_threshhold).astype('float')
                        if list(y_pred)[0] == 0:
                            st.write(f"This client is negatively classified by the model \
                            with the chosen threshold value {selected_threshhold}, so, it is unlikely that he/she will respond to the marketing campagin")
                        
                        else:
                            st.write(f"This client is positively classified by the model \
                            with the chosen threshold value {selected_threshhold}, so, it is likely that he/she will respond to the marketing campagin")
                    
                    else:
                        st.warning("Please Uncheck the evaluation checkbox")
        else:
            st.warning("Select Some Features")

            



    
