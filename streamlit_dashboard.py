import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json
import datetime as dt
import requests
from bs4 import BeautifulSoup
import re
from functools import lru_cache
from utils.helper import *
from typing import Tuple, Optional
from dataclasses import dataclass
from typing import Dict, List
#import hmac

# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if hmac.compare_digest(st.session_state["username"].strip(), os.environ.get("STREAMLIT_USERNAME")) and \
#            hmac.compare_digest(st.session_state["password"].strip(), os.environ.get("STREAMLIT_PASSWORD")):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#             del st.session_state["username"]  # Don't store the username.
#         else:
#             st.session_state["password_correct"] = False

#     # First run or password not correct, show input fields
#     if "password_correct" not in st.session_state:
#         st.text_input("Username", key="username")
#         st.text_input("Password", type="password", key="password")
#         st.button("Login", on_click=password_entered)
#         return False
    
#     # Password correct, return True
#     elif st.session_state["password_correct"]:
#         return True
    
#     # Password incorrect, show input fields
#     else:
#         st.text_input("Username", key="username")
#         st.text_input("Password", type="password", key="password")
#         st.button("Login", on_click=password_entered)
#         st.error("😕 User not authorized")
#         return False

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def wide_space_default():
    st.set_page_config(layout='wide')

wide_space_default()

@st.cache_data
def load_daily_arrival_data():
        return pd.read_csv("commodity_data.csv")
    
@st.cache_data
def load_dca_data():
    df = pd.read_excel('dca_test.xlsx', sheet_name=0).T.reset_index()
    cols = df.iloc[0, 1:]
    index = df.iloc[1:, 0]
    df = df.iloc[1:, 1:]
    df.columns = cols
    df.index = index
    df = df.dropna()
    df.index = pd.to_datetime(df.index, dayfirst = True)
    df = df[~df.index.weekday.isin([5, 6])]
    df = df.resample("W-FRI").mean()

    df_long = df.reset_index().melt(
        id_vars=['index'],
        value_vars=['Rice', 'Wheat', 'Atta(wheat)', 'Gram Dal', 'Tur/Arhar Dal', 'Urad Dal',
                    'Moong Dal', 'Masoor Dal', 'Ground Nut Oil', 'Mustard Oil', 'Vanaspati',
                    'Soya Oil', 'Sunflower Oil', 'Palm Oil', 'Potato', 'Onion', 'Tomato',
                    'Sugar', 'Gur', 'Milk', 'Tea', 'Salt'],
        var_name='Commodity',
        value_name='Price'
    )

    df_long = df_long.rename(columns={'index': 'Date'})
    df_long = df_long.sort_values(['Date', 'Commodity']).reset_index(drop=True)
    df_long['Date'] = pd.to_datetime(df_long['Date'])

    return df_long

# Load and preprocess production data (Cached function)
@st.cache_data
def load_production_data():
    agri_prod = pd.read_excel('dca_data.xlsx', sheet_name='test')
    agri_prod['Crop'] = agri_prod['Crop'].fillna(method='ffill')
    
    # Convert the Year columns to a consistent format
    agri_prod.columns = [str(int(col.split('-')[0]) + 1) if isinstance(col, str) and '-' in col else col for col in agri_prod.columns]
    
    # Convert to long format, including the totals
    agri_prod_long = pd.melt(agri_prod,
                             id_vars=['Crop', 'Season'],
                             var_name='Year',
                             value_name='Value')
    
    # Convert Year column to numeric
    agri_prod_long['Year'] = pd.to_numeric(agri_prod_long['Year'].apply(lambda x: x[:2] + x[-2:]))
    
    return agri_prod_long

# Load and preprocess horticulture data
@st.cache_data
def load_horticulture_data():
    horti_df = pd.read_excel('dca_data.xlsx', sheet_name='horti', header=[0, 1], index_col=[0])
    horti_df = horti_df.reset_index()
    horti_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in horti_df.columns]

    crops_col = horti_df.columns[0]
    horti_long = pd.melt(horti_df, id_vars=[crops_col], var_name='Year_Metric', value_name='Value')
    horti_long[['Year', 'Metric']] = horti_long['Year_Metric'].str.rsplit('_', n=1, expand=True)
    horti_long = horti_long.pivot_table(values='Value', index=[crops_col, 'Year'], columns='Metric', aggfunc='first').reset_index()
    horti_long = horti_long.rename(columns={crops_col: 'Crops', 'Area': 'Area_in_hectares', 'Production': 'Production_in_tonnes'})
    horti_long = horti_long[~horti_long['Crops'].isin(["Fruits", 'Citrus', 'Vegetables'])]
    horti_long['Year'] = horti_long['Year'].apply(lambda x: x.strip()[:2] + x.strip()[-2:])
    horti_long.columns = ['Crops', 'Year', 'Area', 'Production_in_tonnes']
    
    return horti_long

# Load the GeoJSON file for Indian states
@st.cache_data
def load_geojson():
    with open('india_states.geojson', 'r') as f:
        return json.load(f)

# if check_password():
# List of crops
list_of_crops = ['Rice', 'Wheat', 'Maize', 'Barley', 'Jowar', 'Bajra', 'Ragi', 'Small Millets', 
        'Shree Anna /Nutri Cereals', 'Nutri/Coarse Cereals', 'Cereals', 'Tur', 'Gram', 
        'Urad', 'Moong', 'Lentil', 'Other Pulses', 'Total Pulses', 'Total Food Grains', 
        'Groundnut', 'Castorseed', 'Sesamum', 'Nigerseed', 'Soybean', 'Sunflower',
        'Rapeseed & Mustard', 'Linseed', 'Safflower', 'Total Oil Seeds', 'Sugarcane', 
        'Cotton', 'Jute', 'Mesta', 'Jute & Mesta', 'Tobacco', 'Sannhemp', 'Guarseed']

# Load data
df_long = load_dca_data()
agri_prod_long = load_production_data()
horti_long = load_horticulture_data()
india_geojson = load_geojson()
cpi_data = load_cpi_data()

# Get the min and max dates for the slider
min_date = df_long['Date'].min()
max_date = df_long['Date'].max()
three_months_ago = max_date - pd.DateOffset(months=3)
df_long['Days'] = (df_long['Date'] - min_date).dt.days

# Filter the data for the last three months by default
df_long_default = df_long[df_long['Date'] >= three_months_ago]


# Streamlit app
st.title("Inflation Monitoring Dashboard")
col1, col2, col3, col4 = st.columns(4)

cpi_headline_metrics = get_broad_metrics(cpi_data, "All Groups")
cpi_food_metrics = get_broad_metrics(cpi_data, "Food and beverages")
cpi_core_metrics = get_broad_metrics(cpi_data, "Ex Food & Fuel")
cpi_fuel_metrics = get_broad_metrics(cpi_data, "Fuel and light")


with col1:
    custom_metric(label = f"CPI-Headline Y-o-Y for {cpi_headline_metrics['month']}",
                current = f"{cpi_headline_metrics['current mom']}",
                previous = f"{cpi_headline_metrics['previous mom']}")

with col2:
    custom_metric(label = f"CPI-Food Y-o-Y for {cpi_food_metrics['month']}",
                current = f"{cpi_food_metrics['current mom']}",
                previous = f"{cpi_food_metrics['previous mom']}")

with col3:
    custom_metric(label = f"CPI-Core Y-o-Y for {cpi_core_metrics['month']}",
                current = f"{cpi_core_metrics['current mom']}",
                previous = f"{cpi_core_metrics['previous mom']}")
    
with col4:
    custom_metric(label = f"CPI-Fuel Y-o-Y for {cpi_fuel_metrics['month']}",
                current = f"{cpi_fuel_metrics['current mom']}",
                previous = f"{cpi_fuel_metrics['previous mom']}")


# tab1, tab2, tab3, tab4, tab5 = st.tabs(["DCA Retail Price Trends", 
#                                   "Rainfall Deviation", 
#                                   "Agricultural Production Trends", 
#                                   "Arrivals and Wholesale Prices",
#                                   "Daily Arrivals"])

tab1, tab2, tab3, tab5 = st.tabs(["DCA Retail Price Trends", 
                                  "Rainfall Deviation", 
                                  "Food Production Trends",
                                  "Daily Arrivals"])

with tab1:
    st.header("DCA Retail Price Trends")
    url_dca = "https://fcainfoweb.nic.in/reports/report_menu_web.aspx"
    st.write("Data Source: [https://fcainfoweb.nic.in/reports/report_menu_web.aspx](%s)" % url_dca)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Add "Select All" option to the multiselect
        all_commodities_option = ["All Commodities"] + list(df_long['Commodity'].unique())
        commodities_selected = st.multiselect(
            "Select commodities",
            options=all_commodities_option,
            default=['Tomato', 'Potato', 'Onion']
        )
        
        # Handle "All Commodities" selection
        if "All Commodities" in commodities_selected:
            commodities = list(df_long['Commodity'].unique())
        else:
            commodities = commodities_selected

        normalize = st.checkbox("Normalize prices to 100")
        # ... rest of the code remains the same ...

        start_date = st.date_input("Start date", value=three_months_ago, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

        if start_date < three_months_ago.date():
            filtered_df_long = df_long[(df_long['Commodity'].isin(commodities)) &
                                    (df_long['Date'] >= pd.Timestamp(start_date)) &
                                    (df_long['Date'] <= pd.Timestamp(end_date))]
        else:
            filtered_df_long = df_long_default[(df_long_default['Commodity'].isin(commodities)) &
                                            (df_long_default['Date'] >= pd.Timestamp(start_date)) &
                                            (df_long_default['Date'] <= pd.Timestamp(end_date))]

        if normalize:
            normalized_df_list = []
            for commodity in commodities:
                commodity_df = filtered_df_long[filtered_df_long['Commodity'] == commodity].copy()
                starting_price = commodity_df['Price'].iloc[0]
                commodity_df['Price'] = (commodity_df['Price'] / starting_price) * 100
                normalized_df_list.append(commodity_df)
            filtered_df_long = pd.concat(normalized_df_list)

        fig = px.line(filtered_df_long, x='Date', y='Price', color='Commodity',
                    title=f'Price Evolution of {", ".join(commodities)}')
        st.plotly_chart(fig)

    with col2:
        st.subheader("Week-on-Week Momentum (%)")
        latest_5_weeks_df = filtered_df_long.groupby('Commodity').apply(lambda x: x.tail(5)).reset_index(drop=True)
        pct_change_table_data = []
        for commodity in commodities:
            commodity_df = latest_5_weeks_df[latest_5_weeks_df['Commodity'] == commodity].copy()
            commodity_df['Pct_Change'] = commodity_df['Price'].pct_change() * 100
            pct_change_table_data.append(commodity_df.tail(4))

        if pct_change_table_data:
            pct_change_df = pd.concat(pct_change_table_data)[['Commodity', 'Date', 'Pct_Change']].dropna()
            pct_change_df['Date'] = pd.to_datetime(pct_change_df['Date'])
            pct_change_wide = pct_change_df.pivot(index='Commodity', columns='Date', values='Pct_Change')
            
            # Calculate cumulative momentum
            cumulative_momentum = []
            for commodity in pct_change_wide.index:
                momentum_values = pct_change_wide.loc[commodity].values
                cum_momentum = np.prod([(1 + m/100) for m in momentum_values]) - 1
                cumulative_momentum.append(cum_momentum * 100)
                
            pct_change_wide = pct_change_wide.sort_index(axis=1, ascending=True)
            
            num_weeks = len(pct_change_wide.columns)
            column_names = ['Three weeks ago', 'Two weeks ago', 'Previous week', 'Latest week'][-num_weeks:]
            column_dates = pct_change_wide.columns.strftime('%d-%m-%y')
            
            new_column_names = [f'{name}\n({date})' for name, date in zip(column_names, column_dates)]
            pct_change_wide.columns = new_column_names
            
            # Add cumulative momentum column
            pct_change_wide['4-Week\nMomentum (%)'] = cumulative_momentum
            
            pct_change_wide = pct_change_wide.reset_index()
            pct_change_wide = pct_change_wide.round(2)
            
            
            st.markdown(pct_change_wide.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.write("No data available for the selected period and commodities.")
            

with tab2:
    st.header("Rainfall Deviation")
    col1, col2 = st.columns(2)
    rainfall_labels = get_rainfall_labels()
    with col1:
        url = "https://mausam.imd.gov.in/responsive/rainfallinformation_state.php"
        st.write("Data Source: [https://mausam.imd.gov.in/responsive/rainfallinformation_state.php](%s)" % url)
        rainfall_type = st.selectbox(
            "Select Period",
            options=[
                {'label': rainfall_labels[0], 'value': 'D'},
                {'label': rainfall_labels[1], 'value': 'W'},
                {'label': rainfall_labels[2], 'value': 'M'},
                {'label': rainfall_labels[3], 'value': 'C'}
            ],
            format_func=lambda x: x['label'],
            index=3
        )['value']

        df_rain = fetch_rainfall_data(rainfall_type)
        
        fig = px.choropleth(
            df_rain,
            geojson=india_geojson,
            locations='state',
            featureidkey='properties.ST_NM',
            color='deviation',
            hover_name='state',
            hover_data=['actual', 'normal'],
            color_continuous_scale='Geyser_r',
            range_color=[-100, 100],
            title='Rainfall Deviation from Normal'
        )

        fig.update_geos(
            fitbounds="locations",
            visible=False,
            projection_scale=5.5,  # Zoom level of the map
            center={"lat": 20.5937, "lon": 78.9629},  # Centered around India
        )
        fig.update_layout(
            title_font=dict(size=20, family='Arial'),
            title_x=0,
            title_y=0.95,
            margin={"r":0,"t":0,"l":0,"b":0},
            height=500,
            coloraxis_colorbar=dict(
                title="Deviation (%)",
                title_font=dict(size=14, family='Arial')
            )
        )

        st.plotly_chart(fig)

    with col2:
        st.subheader("Major Producers | Statewise")
        url_major = "https://upag.gov.in/"
        st.write("Data Source: [https://upag.gov.in/](%s)" % url_major)
        
        # Add None as the first option
        selected_commodity = st.selectbox("Select a commodity:", ["None"] + list_of_crops)
        
        # Only fetch and display data if a commodity is selected
        if selected_commodity != "None":
            df = fetch_major_producers(selected_commodity)
            if df is not None:
                st.dataframe(df.round(1))
        
    st.subheader("Rainfall Data")
    st.dataframe(df_rain.round(1))

with tab3:
    # st.header("Food Production | Yearly")

    # # Separate the total production data for calculation purposes
    # agri_prod_totals = agri_prod_long[agri_prod_long['Season'] == 'Total']

    # # Filter the data for non-total seasonal production for visualization
    # agri_prod_seasonal = agri_prod_long[agri_prod_long['Season'] != 'Total']

    # # Get data for the latest year
    # latest_year = agri_prod_long['Year'].max()
    # previous_year = latest_year - 1

    # # Calculate total production metrics using the specific "Total Food Production" row
    # total_production_latest = agri_prod_totals[(agri_prod_totals['Crop'] == 'Total Food Production') & 
    #                                         (agri_prod_totals['Year'] == latest_year)]['Value'].sum()
    # total_production_previous = agri_prod_totals[(agri_prod_totals['Crop'] == 'Total Food Production') & 
    #                                             (agri_prod_totals['Year'] == previous_year)]['Value'].sum()
    # total_production_delta = ((total_production_latest - total_production_previous) / total_production_previous) * 100

    # # # Calculate metrics for Cereals, Pulses, and Oilseeds using only "Total" season rows
    # cereal_value, cereal_delta = calculate_group_metrics('Total Cereals', agri_prod_totals)
    # pulse_value, pulse_delta = calculate_group_metrics('Total Pulses', agri_prod_totals)
    # oilseed_value, oilseed_delta = calculate_group_metrics('Oil', agri_prod_totals)

    # # # Create four columns for metrics
    # # col1, col2, col3, col4 = st.columns(4)

    # # # Display metrics in each column
    # # col1.metric(label="Total Production", value=f"{total_production_latest:.0f} lakh MT", delta=f"{total_production_delta:.2f}%")
    # # col2.metric(label="Cereals Production", value=f"{cereal_value:.0f} lakh MT", delta=f"{cereal_delta:.2f}%")
    # # col3.metric(label="Pulses Production", value=f"{pulse_value:.0f} lakh MT", delta=f"{pulse_delta:.2f}%")
    # # col4.metric(label="Oilseeds Production", value=f"{oilseed_value:.0f} lakh MT", delta=f"{oilseed_delta:.2f}%")

    # # Create two columns for plots
    # plot_col1, plot_col2 = st.columns(2)

    # # Plot 1: Dropdown for individual crop selection
    # with plot_col1:
    #     selected_crop = st.selectbox('Select a crop', options=sorted(agri_prod_seasonal['Crop'].unique()), index =18)

    #     # Filter the dataframe based on user selection
    #     filtered_df = agri_prod_seasonal[agri_prod_seasonal['Crop'] == selected_crop]

    #     # Create the stacked bar chart for production trends with seasonal breakdown
    #     fig1 = px.bar(filtered_df,
    #                 x='Year',
    #                 y='Value',
    #                 color='Season',
    #                 labels={'Value': 'Production (in lakh tonnes)'},
    #                 title=f'Production Trend for {selected_crop}',
    #                 height=600)

    #     # Update layout for better readability
    #     fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    #     # Display the chart
    #     st.plotly_chart(fig1, use_container_width=True)

    # # Plot 2: Bar plot for Year-on-Year percentage changes of the selected crop for the last 10 years using agri_prod_totals
    # with plot_col2:
    #     # Filter data for the selected crop using agri_prod_totals
    #     selected_crop_totals_df = agri_prod_totals[agri_prod_totals['Crop'] == selected_crop]

    #     # Calculate the Year-on-Year percentage change
    #     selected_crop_totals_df = selected_crop_totals_df.sort_values(by='Year').reset_index(drop=True)
    #     selected_crop_totals_df['YoY_Change'] = selected_crop_totals_df['Value'].pct_change() * 100

    #     # Filter to only include the last 10 years of data
    #     selected_crop_totals_df_last_10_years = selected_crop_totals_df.tail(10)

    #     # Create a bar plot for Year-on-Year percentage changes for the last 10 years
    #     fig2 = px.bar(
    #         selected_crop_totals_df_last_10_years,
    #         x='Year',
    #         y='YoY_Change',
    #         text=selected_crop_totals_df_last_10_years['YoY_Change'].round(1),
    #         labels={'YoY_Change': 'Y-o-Y Change (%)'},
    #         title=f'Y-o-Y Change (%) in Production for {selected_crop} (Last 10 Years)',
    #         height=600,
    #         color_discrete_sequence=['#1f77b4']
    #     )

    #     # Update the bar plot's layout for better readability
    #     fig2.update_layout(
    #         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #         yaxis_title='Y-o-Y Change (%)',
    #         xaxis_title='Year'
    #     )
    #     fig2.update_traces(textposition='outside')

    #     # Display the chart
    #     st.plotly_chart(fig2, use_container_width=True)

    # # Optional: Display the filtered dataframe
    # if st.checkbox('Show data'):
    #     st.write(filtered_df)
        
    st.header("Horticultural Production | Yearly")
    
    # Display production metrics
    col1, col2, col3 = st.columns(3)
    citrus_latest, citrus_change = get_latest_and_change(horti_long, 'Citrus Total')
    fruits_latest, fruits_change = get_latest_and_change(horti_long, 'Total Fruits')
    veg_latest, veg_change = get_latest_and_change(horti_long, 'Total Vegetables')

    #col1.metric("Citrus Production", f"{citrus_latest:.2f} MT", f"{citrus_change:.2f}")
    col1.metric("Fruits Production", f"{fruits_latest/100000:.2f} lakh MT", f"{fruits_change:.2f}")
    col2.metric("Vegetables Production", f"{veg_latest/10000:.2f} lakh MT", f"{veg_change:.2f}")

    # Create two columns for plots
    plot_col1, plot_col2 = st.columns(2)

    # with plot_col1:
    #     selected_crop = st.selectbox('Select a crop', options=sorted(horti_long['Crops'].unique()))
    #     filtered_df = horti_long[horti_long['Crops'] == selected_crop].sort_values(by='Year')

    #     fig1 = px.bar(filtered_df, x='Year', y='Production (in tonnes)', title=f'Production Trend for {selected_crop}')
    #     fig1.add_scatter(x=filtered_df['Year'], y=filtered_df['Area'], mode='lines+markers', name='Area', yaxis='y2')
    #     fig1.update_layout(yaxis2=dict(title='Area (hectares)', overlaying='y', side='right', showgrid=False))
    #     st.plotly_chart(fig1, use_container_width=True)

with plot_col1:
    # Get the sorted list of crops
    sorted_crops = sorted(horti_long['Crops'].unique())
    # Find the index of 'onion' in the sorted list
    default_index = sorted_crops.index('Onion')
    
    selected_crop = st.selectbox('Select a crop', 
                                options=sorted_crops,
                                index=default_index)
    filtered_df = horti_long[horti_long['Crops'] == selected_crop].sort_values(by='Year')

    fig1 = px.bar(filtered_df, x='Year', y='Production_in_tonnes', title=f'Production Trend for {selected_crop}')
    fig1.add_scatter(x=filtered_df['Year'], y=filtered_df['Area'], mode='lines+markers', name='Area', yaxis='y2')
    fig1.update_layout(yaxis2=dict(title='Area (hectares)', overlaying='y', side='right', showgrid=False))
    st.plotly_chart(fig1, use_container_width=True)

    with plot_col2:
        filtered_df['YoY_Change'] = filtered_df['Production_in_tonnes'].pct_change() * 100
        fig2 = px.bar(filtered_df.tail(10), x='Year', y='YoY_Change', 
                    text=filtered_df.tail(10)['YoY_Change'].round(1),
                    title=f'Y-o-Y Change (%) in Production for {selected_crop}',
                    color_discrete_sequence=['#1f77b4'])
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

# with tab4:
#     commodity_dict = {
#     "Gram": 1, "Groundnut": 3, "Masur/Lentil (Dal/Split)": 5, "Groundnut Oil": 4, "Lentil": 6, "Moong (Dal/Split)": 7, "Moong": 8, "Onion": 9,
#     "Paddy": 10, "Potato": 12, "Rapeseed & Mustard Oil": 14, "Rapeseed & Mustard": 13, "Rice": 15, "Soybean": 16, "Sunflower": 18,
#     "Tur (Dal/Split)": 21, "Tomato": 20, "Tur": 22, "Urad (Dal/Split)": 23, "Urad": 24, "Wheat Atta": 26, "Wheat": 25, "Sesamum": 27,
#     "Avare Dal": 32, "Bajra": 33, "Barley": 34, "Castorseed": 36, "Cotton": 37, "Foxtail Millet": 41, "Cowpea": 40, "Jute": 45,
#     "Guarseed": 43, "Jowar": 44, "Kulthi": 48, "Kodo Millet": 47, "Lakh": 49, "Linseed": 50, "Maize": 53, "Nigerseed": 56, "Peas": 58,
#     "Ragi": 62, "Ramdana": 64, "Safflower": 66, "Sannhemp": 69, "Sugarcane": 72, "Tobacco": 74, "Select":0}
    
#     st.subheader("Arrivals and Wholesale Prices")
#     url_arr = "https://upag.gov.in/dash-reports/pricesmonthcomparison?rtab=Analytics&rtype=dashboards"
#     st.write("Data Source: [https://upag.gov.in/dash-reports/pricesmonthcomparison?rtab=Analytics&rtype=dashboards](%s)" % url_arr)
    
#     # Create dropdown with None as default
#     selected_commodity = st.selectbox(
#         'Select a commodity',
#         options=['Select'] + sorted(commodity_dict.keys()),  # Add None as first option
#         index=0  # Default to None (first item)
#     )

#     # Only fetch and display data if a commodity is selected
#     if selected_commodity != 'Select':
#         # Get the ID of selected commodity
#         selected_commodity_id = commodity_dict[selected_commodity]
        
#         # Show loading message while fetching data
#         with st.spinner(f'Loading data for {selected_commodity}...'):
#             data_object = fetch_and_process_data(
#                 commodity_id=selected_commodity_id,
#                 month_from=1,
#                 year_from=2014,
#                 month_to=12,
#                 year_to=dt.datetime.today().year
#             )
#             data_arr = data_object.dataframe
#             title_arrivals = data_object.title
            
#             # Display the plot
#             plot_comparison(data_arr)
#     else:
#         # Optional: Display a message when no commodity is selected
#         st.write("Please select a commodity to view the data.")

with tab5:
    # Load data
    data = load_daily_arrival_data()
    if data is not None:
        st.write("Data loaded successfully")
    else:
        st.write("Failed to load data")

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y')

    # Title and description
    st.title("Mandi Arrival Analysis")
    st.write("Source: https://agmarknet.gov.in/")

    # Filters section
    st.header("Filters")

    # Commodity selection with capitalized names
    commodity_list = data['Commodity'].unique()
    selected_commodity = st.selectbox(
        "Select Commodity", 
        commodity_list,
        index=35,
        format_func=lambda x: x.capitalize()  # Display names in capitalized format
    )

    # Get current date
    current_date = dt.datetime.now() - dt.timedelta(2)
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day

    # Default date range for seasonality analysis: start of the current month to today
    start_of_current_month = current_date.replace(day=1)  # First day of the current month
    end_date_default = current_date

    # Date range selection for seasonality analysis
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", start_of_current_month)
    with col2:
        end_date = st.date_input("End Date", data['Date'].max(), max_value=data['Date'].max())

    # Filter data based on selections for seasonality analysis
    filtered_data = data[
        (data['Commodity'] == selected_commodity) & 
        (data['Date'] >= pd.to_datetime(start_date)) & 
        (data['Date'] <= pd.to_datetime(end_date))
    ]

    # Checkbox to toggle display of filtered data
    show_data = st.checkbox("Show Filtered Data")

    # Display filtered data if checkbox is checked
    if show_data:
        st.subheader("Filtered Data")
        st.write(data[(data['Commodity'] == selected_commodity)])

    # Create two columns for the charts
    col1, col2 = st.columns(2)

    # Time Series Line Chart: January to December for the last 3 years
    with col1:
        #st.subheader(f"Monthly Arrival Trend for {selected_commodity.capitalize()}")

        # Function to get monthly sums for January to December for the last 3 years
        def get_monthly_sums(data, selected_commodity, current_year, current_month, current_day):
            results = []
            for year in range(current_year - 4, current_year + 1):  # Last 3 years
                for month in range(1, 13):  # January to December
                    if year == current_year and month == current_month:
                        # For the current month, include data only up to the current day
                        monthly_data = data[
                            (data['Commodity'] == selected_commodity) & 
                            (data['Date'].dt.year == year) & 
                            (data['Date'].dt.month == month) & 
                            (data['Date'].dt.day <= end_date.day)
                        ]
                    else:
                        # For other months, include the full month's data
                        monthly_data = data[
                            (data['Commodity'] == selected_commodity) & 
                            (data['Date'].dt.year == year) & 
                            (data['Date'].dt.month == month)
                        ]
                    total_value = monthly_data['Total Value'].sum()/1000
                    if total_value > 0:  # Only include months with data
                        results.append({
                            'Year': year,
                            'Month': month,
                            'Total Value': total_value
                        })
            return pd.DataFrame(results)

        # Get monthly sums for the last 3 years
        monthly_sums = get_monthly_sums(data, selected_commodity, current_year, current_month, current_day)

        # Plot time series line chart for monthly sums
        fig = px.line(
            monthly_sums, 
            x='Month', 
            y='Total Value', 
            color='Year', 
            title=f'Monthly Arrival Trend for {selected_commodity.capitalize()}',
            labels={'Month': 'Month', 'Total Value': 'Total Arrivals (thousand tonnes)'},
            markers=True  # Add markers for better visibility
        )

        # Customize x-axis to show month names
        fig.update_xaxes(
            tickvals=list(range(1, 13)),  # Months 1 to 12
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )

        st.plotly_chart(fig, use_container_width=True)  # Use full width of the column

    # Seasonality Analysis: Bar chart for selected date range (default: start of current month to today)
    with col2:
        #st.subheader(f"Seasonality Analysis")

        # Function to get the same date range for previous years
        def get_previous_years_data(data, start_date, end_date, selected_commodity, years_back):
            """
            Retrieve data for the same date range across previous years.
            """
            results = []
            for i in range(0, years_back + 2):
                prev_start = start_date - pd.DateOffset(years=i)
                prev_end = end_date - pd.DateOffset(years=i)
                prev_data = data[
                    (data['Commodity'] == selected_commodity) & 
                    (data['Date'] >= prev_start) & 
                    (data['Date'] <= prev_end)
                ]
                prev_data['Year'] = prev_start.year
                results.append(prev_data)
            return pd.concat(results)

        # Get data for the past 3 years
        past_years_data = get_previous_years_data(
            data, 
            pd.to_datetime(start_date), 
            pd.to_datetime(end_date), 
            selected_commodity, 
            years_back=3
        )

        # Sum arrivals for each year
        seasonality_summary = past_years_data.groupby('Year')['Total Value'].sum().div(1000).reset_index()

        # Plot bar chart for seasonality
        fig3 = px.bar(
            seasonality_summary, 
            x='Year', 
            y='Total Value', 
            title=f'Monthly arrivals as on {end_date} vis-a-vis previous years',
            color_discrete_sequence=['#9467bd']  # Use the orange color from the Pastel palette
        )

        # Customize x-axis labels to remove .5
        fig3.update_xaxes(
            tickvals=seasonality_summary['Year'],  # Use integer years as tick values
            ticktext=seasonality_summary['Year'].astype(str)  # Display as strings
        )

        # Improve layout and aesthetics
        fig3.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Arrivals (thousand tonnes)",
            showlegend=False,  # Hide legend since colors are for years
            template="plotly_white"  # Use a clean and modern template
        )

        st.plotly_chart(fig3, use_container_width=True)  # Use full width of the column

st.markdown("""
    <p style="font-size: 14px;">
        <strong>Designed and developed by</strong><br>
        <strong><span style="color: coral;">Prices and Monetary Research Division, DEPR</span></strong>
    </p>
""", unsafe_allow_html=True)
