########################################################
# Loading librairies
########################################################

import streamlit as st
import requests
import joblib
#import json
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from PIL import Image
import shap
import matplotlib.pyplot as plt
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.annotations import Label
import plotly.express as px

########################################################
# Session for the API
########################################################
def fetch(session, url):

    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

session = requests.Session()

def client_prediction(id):
    # Getting client's prediction
    response = fetch(session, f"http://api-credit-capacity.herokuapp.com/api/predictions/clients/{id}")
    #http://api-credit-capacity.herokuapp.com/api/predictions/clients/{id}
    #http://127.0.0.1:8000/api/predictions/clients/{id}
    if response:
        return response
    else:
        return "Error"

########################################################
# Loading icon/image
########################################################
image = Image.open("pngwing.com.png")

########################################################
# Loading data, model and preprocessing
########################################################
# Data description of features:
description = pd.read_csv('HomeCredit_columns_description.csv',
                                verbose=False,
                                encoding='ISO-8859-1',
                                )


# main data:

@st.cache
def load_data ():
    data = pd.read_csv('data_try.csv')
    # Un échantillon à tester
    data = data.sample(frac=0.05)
    # Garder les colonnes avec 20 % ou plus de valeurs manquantes
    data = data.dropna(thresh=0.8*len(data), axis=1)
    data.drop(columns = {'Unnamed: 0'}  , inplace = True)
    data = data.replace([np.inf, -np.inf], np.nan)
    #data.fillna(0, inplace=True)
    return data


# Chargement du modèle  
model = joblib.load('LGBM_P7.pkl')
# Chargement du preprocessor
loaded_preprocessor = joblib.load('preprocessor_P7.joblib')

# Preprocessing
data = load_data()
X = data.drop(['SK_ID_CURR'], axis=1)
X_ = loaded_preprocessor.fit_transform(X)
X = pd.DataFrame(X_, index=X.index, columns=X.columns)
X['SK_ID_CURR'] = data['SK_ID_CURR']

########################################################
# Page section (1:"🗺️ Home", 2:"🚩 Client prediction")
########################################################
sb = st.sidebar # defining the sidebar
sb.image(image)

sb.markdown("🚀 **Navigation**")
page_names = ["🗺️ Home", "🚩 Client prediction"]
page = sb.radio("", page_names, index=0)
sb.write('<style>.css-1p2iens { margin-bottom: 0px !important; min-height: 0 !important;}</style>', unsafe_allow_html=True)

msg_sb = sb.info("**How to use it ?** Select the **Home page** to see information related with the project. " \
                "Select **Client prediction** to know whether a specific client will pay the loan based on his information.")

# ---------- Home page ----------    
if page == "🗺️ Home":

    st.subheader("Implement a scoring model")
    st.markdown("This project is part of [OpenClassRooms Data Scientist training](https://openclassrooms.com/fr/paths/164-data-scientist)"\
                " and has two main objectives:")

    objectives_list = '<ul style="list-style-type:disc;">'\
                        '<li>Building a scoring model that will give a prediction about the probability of a client paying the loan.<br>'\
                        'The mision will be treated as a <strong>binary classification problem</strong>.<br>So, 0 will be the class who repaid/pay '\
                        'the loan and 1 will be the class who did not repay/pay the loan.</li>'\
                        '<li>Build an interactive <strong>dashboard for customer relationship managers</strong> to interpret the predictions made by the model,<br>'\
                        'and improve customer knowledge of customer relationship loaders.</li>'\
                    '</ul>'
    st.markdown(objectives_list, unsafe_allow_html=True)
    
    st.subheader("How to use it ?")
    how_to_use_text = "You can navigate through the <strong>Home page</strong> where you will find information related with the project.<br>"\
                        "Also, you can go to the <strong>Client prediction</strong> to know whether a specific client will pay the loan based on his information"
    st.markdown(how_to_use_text, unsafe_allow_html=True)

    st.subheader("Other information")
    other_text = '<ul style="list-style-type:disc;">'\
                    '<li><h4>Data</h4>'\
                    'The data used to develop this project are based on the <a href="https://www.kaggle.com/" target="_blank">Kaggle\'s</a> competition: '\
                    '<a href="https://www.kaggle.com/c/home-credit-default-risk/overview" target="_blank">Home Credit - Default Risk</a></li>'\
                    '<li><h4>Repository</h4>'\
                    'You can find more information about the project\'s code in its <a href="https://github.com/MZMERLI?tab=repositories" target="_blank">Github\' repository</a></li>'\
                '</ul>'
    st.markdown(other_text, unsafe_allow_html=True)

# ---------- Prediction page ----------    
else:

    client_selection_title = '<h3 style="margin-bottom:0; padding: 0.5rem 0px 1rem;">🔎 Client selection</h3>'
    st.markdown(client_selection_title, unsafe_allow_html=True)

    st.write("This WebApps is a decision making helping tool.\n\
    A supervised binary classifier algorithm has been trained in order to predict the client's probability to make a default payment or not.")
    st.warning("Please select a customer on the left sidebar.")

    st.sidebar.header('User Input Values')
    st.subheader('📝 Users data table :')
    
    # ---------- Show clients data ----------
    st.write(data.head(5))

    # ---------- Show data of selected customer ----------
    st.sidebar.header('Select customer number')
    id_client = st.sidebar.selectbox('Identifiant client', data['SK_ID_CURR'])
    client_id = int(id_client)

    st.subheader('📝 Data of selected customer:')

    data_client = data.loc[data['SK_ID_CURR'] == client_id]
    data_client.reset_index()
    st.write(data_client)

    # ---------- Preparation of a comparative graph ----------
    st.subheader('📊 Client\'s total income compared to others')
    amt_inc_total = np.log(data.loc[data['SK_ID_CURR'] == client_id, 'AMT_INCOME_TOTAL'].values[0])
    x_a = [np.log(data['AMT_INCOME_TOTAL'])]
    fig_a = ff.create_distplot(x_a,['AMT_INCOME_TOTAL'], bin_size=0.3)
    fig_a.add_vline(x=amt_inc_total, annotation_text=' Current client!')

    st.plotly_chart(fig_a, use_container_width=True)


    # ---------- More information about the client on sidebar ----------
    data_client= data[data["SK_ID_CURR"]==client_id]
    client_index = data_client.index[0]
    gender = data_client.loc[client_index, "CODE_GENDER"]
    if gender == 1:
        gender = 'Male'
    else:
        gender = 'Female'
    family_status = data_client.loc[client_index, "NAME_FAMILY_STATUS"]
    loan_type = data_client.loc[client_index, "NAME_CONTRACT_TYPE"]
    education = data_client.loc[client_index, "NAME_EDUCATION_TYPE"]
    credit = data_client.loc[client_index, "AMT_CREDIT"]
    annuity = data_client.loc[client_index, "AMT_ANNUITY"]
    fam_members = data_client.loc[client_index, "CNT_FAM_MEMBERS"]
    childs = data_client.loc[client_index, "CNT_CHILDREN"]
    income_type = data_client.loc[client_index, "NAME_INCOME_TYPE"]
    work = income_type
    days_birth = data_client.loc[client_index, "DAYS_BIRTH"]
    age = -int(round(days_birth/365))
    days_employed = data_client.loc[client_index, "DAYS_EMPLOYED"]
    try: 
        years_work = -int(round(days_employed/365))
        if years_work < 1: 
            years_work = 'Less than a year'
        elif years_work == 1:
            years_work = str(years_work) + ' year'
        else: 
            years_work = str(years_work) + ' years'
    except:
        years_work = 'no information'
        
    st.sidebar.subheader('General informations:')
        
    st.sidebar.write('**Gender:** %s' %gender)
    st.sidebar.write('**Age:** %s' %age)
    st.sidebar.write('**Education level:** %s' %education)
    st.sidebar.write('**Marital status:** %s' %family_status)
    st.sidebar.write('**Family members :** %s (including %s children) '%(int(round(fam_members)), int(round(childs))))
    st.sidebar.write('**Work:** %s' %work)
    st.sidebar.write('**Work experiences:** %s ' %years_work)
    st.sidebar.write('')
    st.sidebar.subheader('Credit informations:')
    st.sidebar.write('**Credit amount:** {:,} $'.format(round(credit)))
    st.sidebar.write('**Annuity amount:** {:,} $'.format(round(annuity)))
    st.sidebar.write("---")

    # ---------- DISPLAY Figures   ----------
if page == "🚩 Client prediction":
    placeholder = st.empty()
    info_type = placeholder.radio('⚙️ Do you want more info?:', ['NO  ❌',
                                                              'YES  ✔️',])
    placeholder_bis = st.empty()

    with placeholder_bis.container():
        if info_type == 'YES  ✔️': 

            st.write('Select any information about the client (Univariate analysis):')
            #st.markdown('##')
            selected_features = st.multiselect('', data[data['SK_ID_CURR'] == client_id].dropna(axis=1).select_dtypes('float').columns)
            st.write('Select one second information about the client (Bivariate analysis):')
            #st.markdown('##')
            
            def selectbox_with_default_2(text, values, default="Client's variable: ", sidebar=False):
                func = st.selectbox if sidebar else st.selectbox
                return func(text, np.insert(np.array(values, object), 0, default))
            selected_features_2 = selectbox_with_default_2('', data[data['SK_ID_CURR'] == client_id].dropna(axis=1).columns)
            graph_place = st.empty()
            

            # ---------- Univariate analysis:   ----------

            if selected_features and selected_features_2 == "Client's variable: ":
                for features in selected_features:
                    st.write(features, ': ',description.loc[description['Row'] == features, 'Description'].values[0])
                    with st.container():
                        #col1, col2 = st.columns(2)
                        #with col1:
                        data_client_value = data.loc[data['SK_ID_CURR'] == client_id, features].values

                        # Generate distribution data
                        hist, edges = np.histogram(data.loc[:, features].dropna(), bins=20)
                        hist_source_df = pd.DataFrame({"edges_left": edges[:-1], "edges_right": edges[1:], "hist":hist})
                        max_histogram = hist_source_df["hist"].max()
                        client_line = pd.DataFrame({"x": [data_client_value, data_client_value],
                                                    "y": [0, max_histogram]})
                        hist_source = ColumnDataSource(data=hist_source_df)
                            
                        def plot_feature_distrib(feature_distrib, client_line, hist_source, data_client_value, max_histogram):
                            distrib = figure(title=f"Client value for {feature_distrib} compared to other clients", 
                                             plot_width=1000, plot_height=500)
                            qr = distrib.quad(top="hist", bottom=0, line_color="white", left="edges_left", right="edges_right",
                                fill_color="steelblue", hover_fill_color="orange", alpha=0.5, hover_alpha=1, source=hist_source)

                            distrib.line(x=client_line["x"], y=client_line["y"], line_color="orange", line_width=2, line_dash="dashed")
                            label_client = Label(text="Client's value", x=data_client_value[0], y=max_histogram, text_color="orange",
                                                 x_offset=-50, y_offset=10)
                            hover_tools = HoverTool(tooltips=[("Between:", "@edges_left"), ("and:", "@edges_right"), ("Count:", "@hist")], 
                                                renderers = [qr])
                            distrib.xaxis.axis_label = feature_distrib
                            distrib.y_range.start = 0
                            distrib.y_range.range_padding = 0.2
                            distrib.yaxis.axis_label = "Number of clients"
                            distrib.grid.grid_line_color="grey"
                            distrib.xgrid.grid_line_color=None
                            distrib.ygrid.grid_line_alpha=0.5
                            distrib.add_tools(hover_tools)
                            distrib.add_layout(label_client)
                            return distrib
                            
                        plot = plot_feature_distrib(features,
                                                    client_line,
                                                    hist_source,
                                                    data_client_value,
                                                    max_histogram)
                        st.bokeh_chart(plot, use_container_width=True)
            
            
            # ---------- Bivariate analysis:   ----------

            elif selected_features and selected_features_2 != "Client's variable: ": 
                with graph_place.container():
                    for features in selected_features:
                        
                        # Two continuous variables (scatter plot):
                        if selected_features_2 in data.select_dtypes('float').columns.to_list():
                            data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                            data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                            fig = px.scatter(data, x=features, y=selected_features_2, height=580, opacity=.3)
                            fig.add_trace(go.Scattergl(x=data_client_value_1,
                                                    y=data_client_value_2,
                                                    mode='markers',
                                                    marker=dict(size=10, color = 'red'),
                                                    name='client'))
                            fig.update_layout(legend=dict(
                                                yanchor="top",
                                                y=1,
                                                xanchor="left",
                                                x=1)) 
                            st.plotly_chart(fig, use_container_width=True)
                            st.write('---')
                            
                         # At least one discrete variable (box plot):
                        else:
                            data_client_value_1 = data.loc[data['SK_ID_CURR'] == client_id, features].values
                            data_client_value_2 = data.loc[data['SK_ID_CURR'] == client_id, selected_features_2].values
                            fig = px.box(data, x=selected_features_2, y=features, points="outliers", color=selected_features_2, height=580)
                            fig.update_traces(quartilemethod="inclusive")
                            fig.add_trace(go.Scatter(x=data_client_value_2,
                                                     y=data_client_value_1,
                                                     mode='markers',
                                                     marker=dict(size=10),
                                                     showlegend=False,
                                                     name='client'))
                            st.plotly_chart(fig, use_container_width=True)
                            st.write('---')   
    
    

    # ---------- DISPLAY RESULTS   ----------
st.markdown('##')
st.markdown('##')  
if page == "🚩 Client prediction":
    st.header('''Credit application result''')
    placeholder = st.empty()
    info_type = placeholder.radio('⚙️ Check credit score:', ['Later ❌',
                                                          'Now  ✔️',])
    placeholder_bis2 = st.empty()
    with placeholder_bis2.container():
        if info_type == 'Now  ✔️': 
            
            placeholder = st.empty()
            placeholder.write(f"You've selected client #{client_id}.")
        
            # ---------- Prediction from selected customer data ----------
            df = X.loc[data['SK_ID_CURR'] == client_id]
            df = df.drop(['SK_ID_CURR'], axis=1)
            per_pos = model.predict_proba(df)[0][1]
            
            prediction = client_prediction(client_id)
            per_pos2 = float(prediction["probability1"])
            
            # ---------- plot gauge ----------
            def plot_gauge(prediction_default):
                fig_gauge = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = prediction_default,
                mode = "gauge+number",
                title = {'text': "Risk of default (%)"},
                gauge = {'axis': {'range': [None, 100],
                                'tick0': 0,
                                'dtick':10},
                        'bar': {'color': 'blue',
                                'thickness': 0.3,
                                'line': {'color': 'black'}},
                        'steps' : [{'range': [0, 29.8], 'color': "green"},
                                    {'range': [30.2, 49.8], 'color': "orange"},
                                    {'range': [50.2, 100], 'color': "red"}]}        
                                    ))
        
                fig_gauge.update_layout(width=600, 
                                        height=400,
                                        margin= {'l': 30, 'r': 30, 'b': 30, 't':30})
                return fig_gauge
        
            # ---------- Show results   ----------
            placeholder = st.empty()
        
            with placeholder.container():
                left_column_recom, right_column_recom = st.columns(2)
        
                with left_column_recom:
                    gauge_place = st.empty()
                    for value in np.arange(0, round(per_pos2*100, 1) +.1, step=.1):
                        fig_gauge = plot_gauge(value)
                        gauge_place.plotly_chart(fig_gauge, use_container_width=True)
        
                with right_column_recom: 
                    # Display clients data and prediction
                    st.write(f"Client #{client_id} has **{per_pos2*100:.2f} % of risk** to make default.")
                    st.markdown('##')
        
                    if per_pos2*100 < 30:
                        st.success(f"We recommand to **accept** client's application to loan 😇.")
                    elif (per_pos2*100 >= 30) & (per_pos2*100 <= 50):
                        st.warning(f"Client's chances to make default are between 30 and 50% . We recommand \
                                   to **analyse closely** the data to make your decision 🤐.")
                    else:
                        st.error(f"We recommand to **reject** client's application to loan 😑.")
                        
                    st.markdown('##')
                    st.markdown('##')
                    st.caption(f'''Below 30% of default risk, we recommand to accept client application.\
                               Above 50% of default risk, we recommand to reject client application. \
                                   Between 30 and 50%, your expertise will be your best advice in your decision making.\
                                You can use the "client more informations" page to help in the evaluation.''')
            
            
            # ---------- Display shap explainer of client's prediction   ----------
            X = X.drop(['SK_ID_CURR'], axis=1)

            st.subheader('Prediction explanation')
            placeholder = st.empty()
            placeholder.write(f"Feature importance.")
            
            SHAP_explainer = shap.TreeExplainer(model)
            shap_vals = SHAP_explainer.shap_values(X_)
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(4, 2))
            graph_1=shap.summary_plot(shap_vals, X)
            st.set_option('deprecation.showPyplotGlobalUse', False) # avoid an error
            st.pyplot(graph_1, use_container_width=True)
            #st.bokeh_chart(graph_1, use_container_width=True)

            # Bar plot
            fig, ax = plt.subplots(figsize=(4, 3))
            graph_2=shap.bar_plot(shap_vals[1][1], X)
            st.set_option('deprecation.showPyplotGlobalUse', False) # avoid an error
            st.pyplot(graph_2, use_container_width=True)
            #st.bokeh_chart(graph_2, use_container_width=True)
            st.write('Red color indicates features that are pushing the prediction higher, and red color indicates just the opposite.')

    
########################################################
# Bottom
########################################################
st.write("---")

col_about, col_FAQ, col_doc, col_contact = st.columns(4)

with col_about:
    st.write("About us")

with col_FAQ:
    st.write("FAQ")

with col_doc:
    st.write("Technical documentation")

with col_contact:
    st.write("Contact")
