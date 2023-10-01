import kiwisolver
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib
from os.path import dirname, join, realpath
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Load the saved machine learning model
# loaded_file = joblib.load('models')
loaded_watoto = pickle.load(open('models/Watoto fund predictor.pkl', 'rb'))
loaded_jikimu = pickle.load(open('models/Jikimu fund predictor.pkl', 'rb'))
loaded_bond = pickle.load(open('models/Bond fund predictor.pkl', 'rb'))
loaded_liquid = pickle.load(open('models/Liquid fund predictor.pkl', 'rb'))
loaded_umoja = pickle.load(open('models/Umoja fund predictor.pkl', 'rb'))
loaded_wekeza = pickle.load(
    open('models/Wekeza Maisha fund predictor.pkl', 'rb'))

loaded_fund = None

data = pd.read_csv('NAV/Datasets/UTTAMIS-Datasets.csv')
data['Date_Valued1'] = pd.to_datetime(data['Date_Valued'])
df1 = pd.read_csv('NAV Dataset.csv')
df2 = pd.read_csv('Clean_Nav.csv')

st.set_page_config(
    page_title="NAV Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",

)


# Creating  a sidebar for navigation
# st.sidebar.header('NET ASSET VALUE FORECAST')
custom_sidebar_css = """
<style>
/* Add styles to the sidebar header */
.sidebar-header {
    border-bottom: 2px solid #000; /* Add an underline to the sidebar header */
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-color : #fff;
    color: #fff;
}
.css-1v0mbdj.e115fcil1{
    margin-left: 85px;
    border-radius: 45px;
    border-color: blue;
    padding-botom: 400px;
}
.class="css-k7vsyb.e1nzilvr2{
    font-size: 5px;
    color: #fff;
    text-align: center;  
}

.css-1v0mbdj-e115fcil1{
   margin-left: 650;
}

.nsewdrag.drag{
margin-right: 700px
}

</style>
"""


# Inject the custom CSS style for the sidebar header
st.markdown(custom_sidebar_css, unsafe_allow_html=True)

# Create a sidebar for navigation
st.sidebar.image('images/growth.png', width=150)
st.sidebar.markdown(
    '<center><h1 class="sidebar-header" style="color: #fff;"> UTT AMIS  NAV FORECASTING</h1></center>', unsafe_allow_html=True)
# Rest of your sidebar content...

page_selection = st.sidebar.radio('Select a Page', [
                                  'Home', 'Dataset Viewer', 'NAV Trend Yearly', 'Visualization', 'Predictions', 'Investment Calculator'])

# Creating Pages
pages = {
    'Home': '',
    'Dataset Viewer': 'View and filter the dataset',
    'NAV Trend Yearly': 'View The Trend Of NAV Over Specific Year',
    'Predictions': 'Welcome To Predict The Future',
    'Visualization': 'Welcome To Visualize And Get Insights With Us',
    'Investment Calculator': 'Calculate With Us To Know Your Future Profit',

}


header_container = st.container()

# Define a custom CSS style to center-align the content
custom_css = """
<style>
/* Center-align the content */
.header-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #A28089;
}

/* Set plot label colors to black */
.plot-label {
    fill: black !important;
}

/* Remove default Plotly legend title */
.legendtitle {
    font-size: 0;
}

.css-1cf1cqr.ea3mdgi1{
 display: none;
 visibility: hidden;
}
</style>
"""

# Inject the custom CSS style
st.markdown(custom_css, unsafe_allow_html=True)

# Add the logo and centered title to the header container
with header_container:
    st.markdown('<div class="header-content"><h1 style="color: #A28089;"> 📊 UTT AMIS NET ASSET VALUE FORECASTING</h1></div>', unsafe_allow_html=True)

# Home Page
if page_selection == 'Home':
    st.markdown('<div align="center"><h2 style="color: #A28089;>' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)

    st.markdown('<div align="center"><h2 style="color: #A28089;">UTT AMIS FUNDS</h2></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="display: flex; justify-content: space-between;">',
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    images = {
        "UMOJA": 'images/umoja-fund-logo.gif',
        "WATOTO": 'images/watoto1.jpeg',
        "LIQUID": 'images/liquid_logo.jpeg',
        "WEKEZA": 'images/wekeza maisha.jpeg',
        "JIKIMU": 'images/jikimu logo.jpeg',
        "BOND": 'images/bondfund.jpeg',

    }

    with col1:
        st.image(images["UMOJA"], width=80, caption="")

    with col2:
        st.image(images["LIQUID"], width=80, caption="")

    with col3:
        st.image(images["WATOTO"], width=80, caption="")

    with col4:
        st.image(images["WEKEZA"], width=80, caption="")

    with col5:
        st.image(images["JIKIMU"], width=80, caption="")

    with col6:
        st.image(images["BOND"], width=80, caption="")

    st.markdown('<div align="center"><h2 style="color: #A28089;">' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)

    st.markdown('<hr style="height:5px; border:none; color:#A28089; background-color:#A28089;">',
                unsafe_allow_html=True)


# Define the CSS styles for the centered text and paragraph
    style = """
    <style>
    .border-box {
        border: 2px solid #007acc;
        border-radius: 10px;
        padding: 20px;
    }
    .hero-text {
        color: #A28089;
        font-size: 18px;
        text-align: center;
    }
    .hero-paragraph {
        color:#000 ;
        font-size: 16px;
        text-align: center;
    }

    strong{
      color: #A28089;
    }

    .hero-paragraph.h6{
       color:#A28089;
    }
    </style>
"""

# Add the CSS styles
    st.markdown(style, unsafe_allow_html=True)

    st.markdown('<center><h5 class="hero-text">Empowering Your Financial Future Through Predictive Insights At UTT AMIS NAV FORECASTING!</h5></center>', unsafe_allow_html=True)
    st.markdown('<center><p class="hero-paragraph">Welcome to <strong>UTT AMIS NAV FORECASTING</strong>, revolutionizing Tanzanian financial asset management. We provide cutting-edge tech and expert insights for confident investing.</p></center>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    style = """
        <style>
        .border-box {
            border: 2px solid #007acc;
            border-radius: 10px;
            padding: 20px;
        }
        .hero-text {
            color: #A28089;
            font-size: 18px;
            text-align: center;
        }
        .hero-paragraph {
            color: #333333;
            font-size: 16px;
            text-align: center;
        }
        .white-h6 {
            color: #A28089;
            font-size: 14px;
            margin-left: -60px;
        }
        </style>
    """

    # Add the CSS styles
    st.markdown(style, unsafe_allow_html=True)

    col8, col9, col10, col11 = st.columns(4)

    # Inside the column elements:
    with col8:
        st.image('images/dataset.png', width=40)
        st.markdown(
            '<center><h6 class="white-h6"> DATASET VIEWER  </h6></center>',  unsafe_allow_html=True)

    with col9:
        st.image('images/NAV.png', width=40)
        st.markdown(
            '<center><h6 class="white-h6"> DATA ANALYTICS </h6></center>',  unsafe_allow_html=True)

    with col10:
        st.image('images/statistics.png', width=40)
        st.markdown(
            '<center><h6 class="white-h6"> DATA VISUALIZATION </h6></center>',  unsafe_allow_html=True)

    with col11:
        st.image('images/predictive-chart.png', width=40)
        st.markdown(
            '<center><h6 class="white-h6"> FORECASTING</h6></center>',  unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<center> 2023 ©️ dLab Tanzania Group 2 ™️ </center>',
                unsafe_allow_html=True)


elif page_selection == 'Dataset Viewer':
    st.markdown('<div align="center"><h2>' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)
    year_to_display = st.sidebar.selectbox(
        'Choose a Year', data['Date_Valued'].str[-4:].unique())
    filtered_data = data[data['Date_Valued'].str[-4:] == year_to_display]
    st.subheader(f' 📋 Dataset for {year_to_display}')
    st.dataframe(filtered_data)

elif page_selection == 'NAV Trend Yearly':
    st.markdown('<div align="center"><h2>' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)
    st.markdown('<hr style="height:5px; border:none; color:#A28089; background-color:#A28089;">',
                unsafe_allow_html=True)

    # User selects the year
    year_to_display = st.sidebar.selectbox(
        'Choose a Year', data['Date_Valued'].str[-4:].unique())

    # User selects the scheme_Name
    scheme_name_to_display = st.sidebar.selectbox(
        'Choose a Scheme', data['Scheme_Name'].unique())

    # Filter the data based on the selected year and scheme_Name
    filtered_data = data[(data['Date_Valued'].str[-4:] == year_to_display)
                         & (data['Scheme_Name'] == scheme_name_to_display)]

    # Create a centered trend plot with increased size
    if not filtered_data.empty:
        fig = px.line(filtered_data, x='Date_Valued', y='Nav_Per_Unit', color='Scheme_Name',
                      title=f'NAV for {year_to_display} - {scheme_name_to_display}')
        fig.update_layout(
            width=1000,
            height=600,
            margin=dict(autoexpand=True),
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected year and scheme.")


elif page_selection == 'Visualization':
    st.markdown('<div align="center"><h2 style="color: #A28089;>' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)
    st.markdown('<hr style="height:5px; border:none; color:#A28089; background-color:#A28089;">',
                unsafe_allow_html=True)
    st.markdown('<div align="center"><h2 style="color: #A28089;">Avarage NAV For Each Fund</h2></div>',
                unsafe_allow_html=True)
    year_to_display = st.sidebar.selectbox(
        'Choose a Year (Bar Plot)', data['Date_Valued'].str[-4:].unique())
    filtered_data = data[data['Date_Valued'].str[-4:] == year_to_display]
    # Creating a subplot with two vertical bar charts (NAV per unit and Total NAV)
    fig = go.Figure()

    # Bar chart for NAV per unit
    scheme_nav_per_unit = filtered_data.groupby(
        'Scheme_Name')['Nav_Per_Unit'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=scheme_nav_per_unit['Scheme_Name'],
        y=scheme_nav_per_unit['Nav_Per_Unit'],
        name='NAV per Unit',
        marker_color='blue',
    ))

    # # Bar chart for Total NAV
    # scheme_nav_total = filtered_data.groupby('Scheme_Name')['Net_Asset_Value'].sum().reset_index()
    # fig.add_trace(go.Bar(
    #     x=scheme_nav_total['Scheme_Name'],
    #     y=scheme_nav_total['Net_Asset_Value'],
    #     name='Total NAV',
    #     marker_color='green',
    # ))

    fig.update_layout(
        barmode='group',
        title=f'NAV per Unit and Total NAV for FUnds in {year_to_display}',
        xaxis_title='Scheme Name',
        yaxis_title='Value',
        width=800,
        height=600,
        margin=dict(autoexpand=True),
    )

    st.plotly_chart(fig)

    df_grouped = df2.groupby('Scheme Name')[
        ['Sale Price/Unit', 'Repurchase Price/Unit']].mean()
    st.markdown('<div align="center"><h2 style="color: #A28089;">Avarage Fund Perfomance</h2></div>',
                unsafe_allow_html=True)

    # Add a slider to select the year
    selected_year = st.slider('Select a Year', min_value=df2['Year'].min(
    ), max_value=df2['Year'].max(), value=df2['Year'].min())

    # Filter the DataFrame based on the selected year
    filtered_df = df2[df2['Year'] == selected_year]

    # Group the filtered DataFrame by 'Scheme Name' and calculate the mean of 'Sale Price/Unit' and 'Repurchase Price/Unit'
    df_grouped = filtered_df.groupby('Scheme Name')[
        ['Sale Price/Unit', 'Repurchase Price/Unit']].mean().reset_index()

    # Create the bar plot using Plotly Express
    bar_fig = px.bar(
        df_grouped,
        x='Scheme Name',
        y=['Sale Price/Unit', 'Repurchase Price/Unit'],
        title='Average Fund Performance by Fund',
        labels={'value': 'Average Value'},
        barmode='group'
    )
    # Display the bar plot
    st.plotly_chart(bar_fig)

    st.markdown('<div align="center"><h2 style="color: #A28089;">Pie Chart For Distribution of Funds</h2></div>',
                unsafe_allow_html=True)
    year_to_display = st.sidebar.selectbox(
        'Choose a Year (Pie Chart)', data['Date_Valued'].str[-4:].unique())
    filtered_data = data[data['Date_Valued'].str[-4:] == year_to_display]
    st.write(f'Pie Chart for Funds in {year_to_display}')
    scheme_nav_totals = filtered_data.groupby(
        'Scheme_Name')['Net_Asset_Value'].sum().reset_index()

    fig = go.Figure(data=[go.Pie(
        labels=scheme_nav_totals['Scheme_Name'],
        values=scheme_nav_totals['Net_Asset_Value'],
        pull=[0.05, 0.05, 0.05, 0.05],
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        hole=0.3,
    )])

    fig.update_layout(
        width=800,
        height=600,
        scene=dict(aspectmode="cube")
    )

    st.plotly_chart(fig)

    st.markdown('<div align="center"><h2 style="color: #A28089;">Corelation Heatmap Matrix</h2></div>',
                unsafe_allow_html=True)

    correlation_matrix = filtered_df[[
        'Sale Price/Unit', 'Repurchase Price/Unit', 'Units', 'NAV']].corr()

    # Create the heatmap using Plotly Express with 'Viridis' colorscale for better visibility
    heatmap_fig = px.imshow(
        correlation_matrix,
        labels=dict(x='Metrics', y='Metrics', color='Correlation'),
        x=['Sale Price/Unit', 'Repurchase Price/Unit', 'Units', 'NAV'],
        y=['Sale Price/Unit', 'Repurchase Price/Unit', 'Units', 'NAV'],
        color_continuous_scale='Cividis'  # Use 'Viridis' colorscale
    )

    # Display the heatmap
    st.plotly_chart(heatmap_fig)

    funds = ['Umoja Fund', 'Watoto Fund', 'Wekeza Maisha Fund',
             'Jikimu Fund', 'Liquid Fund', 'Bond Fund']

    # Streamlit app
    st.markdown('<div align="center"><h2 style="color: #A28089;">Sale Price Distribution Over Year</h2></div>',
                unsafe_allow_html=True)

    # Iterate through funds and create box plots
    for fund in funds:
        # Filter data for the current fund
        filtered_data = df2[df2['Scheme Name'] == fund]

        # Create a box plot using Plotly Express
        fig = px.box(
            filtered_data,
            x='Year',
            y='Sale Price/Unit',
            title=f'Sale Price Distribution Over Years - {fund}',
            labels={'Sale Price/Unit': 'Price'},
            color='Year'
        )

        # Display the box plot
        st.plotly_chart(fig)

    # List of funds
    funds = ['Umoja Fund', 'Watoto Fund', 'Wekeza Maisha Fund',
             'Jikimu Fund', 'Liquid Fund', 'Bond Fund']

    # Streamlit app
    st.markdown('<div align="center"><h2 style="color: #A28089;">Units Distribution Over Year</h2></div>',
                unsafe_allow_html=True)

    # Iterate through funds and create box plots
    for fund in funds:

        # Filter data for the current fund
        filtered_data = df2[df2['Scheme Name'] == fund]

        # Create a box plot using Plotly Express
        fig = px.box(
            filtered_data,
            x='Year',
            y='Repurchase Price/Unit',
            title=f'Purchase Price Distribution Over Years - {fund}',
            labels={'Units': 'Units'},
            color='Year',
            log_y=True
        )

        # Display the box plot
        st.plotly_chart(fig)

elif page_selection == 'Predictions':
    st.markdown('<div align="center"><h2 style="color: #A28089;">' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    images = {
        "UMOJA": 'images/umoja-fund-logo.gif',
        "WATOTO": 'images/watoto1.jpeg',
        "LIQUID": 'images/liquid_logo.jpeg',
        "WEKEZA": 'images/wekeza maisha.jpeg',
        "JIKIMU": 'images/jikimu logo.jpeg',
        "BOND": 'images/bondfund.jpeg',

    }

    with col1:
        st.image(images["UMOJA"], width=100, caption="")

    with col2:
        st.image(images["LIQUID"], width=100, caption="")

    with col3:
        st.image(images["WATOTO"], width=100, caption="")

    with col4:
        st.image(images["WEKEZA"], width=100, caption="")

    with col5:
        st.image(images["JIKIMU"], width=100, caption="")

    with col6:
        st.image(images["BOND"], width=100, caption="")

    st.markdown('<hr style="height:5px; border:none; color:#A28089; background-color:#A28089;">',
                unsafe_allow_html=True)

    st.markdown('<div align="left"><h5 style="color: #A28089;">Enter the details for prediction</h5></div>',
                unsafe_allow_html=True)
    unique_scheme_names = data['Scheme_Name'].unique()
    # Input fields
    scheme_name = st.selectbox('Select Scheme Name', unique_scheme_names)
    x = st.number_input('Enter day', min_value=1, max_value=31)
    y = st.number_input('Enter month', min_value=1, max_value=12)
    z = st.number_input('Enter Year', min_value=2016, max_value=2024)
    date = f'{z}-{y:02d}-{x:02d}'

    # Predict button
    if st.button('Predict'):
        if scheme_name == 'Watoto Fund':
            loaded_fund = loaded_watoto

        elif scheme_name == 'Jikimu Fund':
            loaded_fund = loaded_jikimu

        elif scheme_name == 'Bond Fund':
            loaded_fund = loaded_bond

        elif scheme_name == 'Liquid Fund':
            loaded_fund = loaded_liquid

        elif scheme_name == 'Umoja Fund':
            loaded_fund = loaded_umoja

        elif scheme_name == 'Wekeza Maisha Fund ':
            loaded_fund = loaded_wekeza

    if loaded_fund is not None:
        with st.spinner("predicting..."):
            # Change the date as needed
            specific_date = pd.DataFrame({'ds': [date]})
            forecast = loaded_fund.predict(specific_date)
            specific_date_prediction = forecast.loc[0, 'yhat']
            st.success(
                f'Predicted Net Asset Value per unit:{specific_date_prediction:.2f} TSH')
elif page_selection == 'Predictions':
    st.markdown('<div align="center"><h2>' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)
    st.subheader('Enter the details for prediction:')
    unique_scheme_names = data['Scheme_Name'].unique()
    # Input fields
    scheme_name = st.selectbox('Select Scheme Name', unique_scheme_names)
    x = st.number_input('Enter day', min_value=1, max_value=31)
    y = st.number_input('Enter month', min_value=1, max_value=12)
    z = st.number_input('Enter Year', min_value=2016, max_value=2024)
    date = f'{z}-{y:02d}-{x:02d}'

    # Predict button
    if st.button('Predict'):
        if scheme_name == 'Watoto Fund':
            loaded_fund = loaded_watoto

        elif scheme_name == 'Jikimu Fund':
            loaded_fund = loaded_jikimu

        elif scheme_name == 'Bond Fund':
            loaded_fund = loaded_bond

        elif scheme_name == 'Liquid Fund':
            loaded_fund = loaded_liquid

        elif scheme_name == 'Umoja Fund':
            loaded_fund = loaded_umoja

        elif scheme_name == 'Wekeza Maisha Fund ':
            loaded_fund = loaded_wekeza

    if loaded_fund is not None:
        with st.spinner("predicting..."):
            # Change the date as needed
            specific_date = pd.DataFrame({'ds': [date]})
            forecast = loaded_fund.predict(specific_date)
            specific_date_prediction = forecast.loc[0, 'yhat']
            st.success(
                f'Predicted Net Asset Value per unit:{specific_date_prediction:.2f} TSH')


elif page_selection == 'Investment Calculator':
    st.markdown('<div align="center"><h2>' +
                pages[page_selection] + '</h2></div>', unsafe_allow_html=True)
    unique_scheme_names = data['Scheme_Name'].unique()
    # Input fields
    scheme_name = st.selectbox('Select fund', unique_scheme_names)
    a = st.number_input('Number of units', min_value=1, max_value=1000)
    d1 = st.date_input("Enter start date")
    d2 = st.date_input("Enter end date")
    d1 = d1.strftime('%d-%m-%Y')
    d2 = d2.strftime('%d-%m-%Y')

    d1 = str(d1)
    d2 = str(d2)

    d1 = d1.replace('/', '-')
    d2 = d2.replace('/', '-')

    # Predict button
    if st.button('Predict'):
        if scheme_name == 'Watoto Fund':
            loaded_fund = loaded_watoto

        elif scheme_name == 'Jikimu Fund':
            loaded_fund = loaded_jikimu

        elif scheme_name == 'Bond Fund':
            loaded_fund = loaded_bond

        elif scheme_name == 'Liquid Fund':
            loaded_fund = loaded_liquid

        elif scheme_name == 'Umoja Fund':
            loaded_fund = loaded_umoja

        elif scheme_name == 'Wekeza Maisha Fund ':
            loaded_fund = loaded_wekeza

    if loaded_fund is not None:
        with st.spinner("predicting..."):
            specific_date = pd.DataFrame({'ds': [d2]})
            initial_date = pd.DataFrame({'ds': [d1]})
            forecast1 = loaded_fund.predict(specific_date)
            forecast2 = loaded_fund.predict(initial_date)
            specific_date_prediction = forecast1.loc[0, 'yhat']
            real_date_value = forecast2.loc[0, 'yhat']
            profit = a*(specific_date_prediction-real_date_value)
            st.success(f'Profit due  {d2}  is :{profit:.2f} Tsh')
            st.success(
                f'Total Amount due  {d2}  is :{specific_date_prediction*a:.2f} Tsh')

# footer
    st.markdown('---')
    st.markdown('<center> 2023 ©️ dLab Tanzania Group 2 ™️ </center>',
                unsafe_allow_html=True)
