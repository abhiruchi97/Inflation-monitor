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

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def wide_space_default():
    st.set_page_config(layout='wide')

wide_space_default()

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


tab1, tab2, tab3, tab4 = st.tabs(["DCA Retail Price Trends", "Rainfall Deviation", "Agricultural Production Trends", "Wholesale Prices and Arrivals"])

with tab1:
    st.header("DCA Retail Price Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        commodities = st.multiselect(
            "Select commodities",
            options=df_long['Commodity'].unique(),
            default=['Tomato', 'Potato', 'Onion']
        )

        normalize = st.checkbox("Normalize prices to 100")

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
            
            pct_change_wide = pct_change_wide.sort_index(axis=1, ascending=True)
            
            num_weeks = len(pct_change_wide.columns)
            column_names = ['Three weeks ago', 'Two weeks ago', 'Previous week', 'Latest week'][-num_weeks:]
            column_dates = pct_change_wide.columns.strftime('%d-%m-%y')
            
            new_column_names = [f'{name}\n({date})' for name, date in zip(column_names, column_dates)]
            pct_change_wide.columns = new_column_names
            
            pct_change_wide = pct_change_wide.reset_index()
            pct_change_wide = pct_change_wide.round(2)
            
            st.markdown(pct_change_wide.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.write("No data available for the selected period and commodities.")
            

with tab2:
    st.header("Rainfall Deviation")
    col1, col2 = st.columns(2)
    
    with col1:
        url = "https://mausam.imd.gov.in/responsive/rainfallinformation_state.php"
        st.write("Data Source: [https://mausam.imd.gov.in/responsive/rainfallinformation_state.php](%s)" % url)
        rainfall_type = st.selectbox(
            "Select Period",
            options=[
                {'label': 'Daily', 'value': 'D'},
                {'label': 'Weekly', 'value': 'W'},
                {'label': 'Monthly', 'value': 'M'},
                {'label': 'Cumulative', 'value': 'C'}
            ],
            format_func=lambda x: x['label'],
            index=2
        )['value']

        df = fetch_rainfall_data(rainfall_type)
        
        fig = px.choropleth(
            df,
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
        st.subheader("Major Commodity Producers")
        
        # Add None as the first option
        selected_commodity = st.selectbox("Select a commodity:", ["None"] + list_of_crops)
        
        # Only fetch and display data if a commodity is selected
        if selected_commodity != "None":
            df = fetch_major_producers(selected_commodity)
            if df is not None:
                st.dataframe(df.round(1))
        
    st.subheader("Rainfall Data")
    st.dataframe(df)

with tab3:
    st.header("Food Production | Yearly")

    # Separate the total production data for calculation purposes
    agri_prod_totals = agri_prod_long[agri_prod_long['Season'] == 'Total']

    # Filter the data for non-total seasonal production for visualization
    agri_prod_seasonal = agri_prod_long[agri_prod_long['Season'] != 'Total']

    # Get data for the latest year
    latest_year = agri_prod_long['Year'].max()
    previous_year = latest_year - 1

    # Calculate total production metrics using the specific "Total Food Production" row
    total_production_latest = agri_prod_totals[(agri_prod_totals['Crop'] == 'Total Food Production') & 
                                               (agri_prod_totals['Year'] == latest_year)]['Value'].sum()
    total_production_previous = agri_prod_totals[(agri_prod_totals['Crop'] == 'Total Food Production') & 
                                                 (agri_prod_totals['Year'] == previous_year)]['Value'].sum()
    total_production_delta = ((total_production_latest - total_production_previous) / total_production_previous) * 100

    # Calculate metrics for Cereals, Pulses, and Oilseeds using only "Total" season rows
    cereal_value, cereal_delta = calculate_group_metrics('Total Cereals', agri_prod_totals)
    pulse_value, pulse_delta = calculate_group_metrics('Total Pulses', agri_prod_totals)
    oilseed_value, oilseed_delta = calculate_group_metrics('Oil', agri_prod_totals)

    # Create four columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    # Display metrics in each column
    col1.metric(label="Total Production", value=f"{total_production_latest:.0f} lakh MT", delta=f"{total_production_delta:.2f}%")
    col2.metric(label="Cereals Production", value=f"{cereal_value:.0f} lakh MT", delta=f"{cereal_delta:.2f}%")
    col3.metric(label="Pulses Production", value=f"{pulse_value:.0f} lakh MT", delta=f"{pulse_delta:.2f}%")
    col4.metric(label="Oilseeds Production", value=f"{oilseed_value:.0f} lakh MT", delta=f"{oilseed_delta:.2f}%")

    # Create two columns for plots
    plot_col1, plot_col2 = st.columns(2)

    # Plot 1: Dropdown for individual crop selection
    with plot_col1:
        selected_crop = st.selectbox('Select a crop', options=sorted(agri_prod_seasonal['Crop'].unique()), index =18)

        # Filter the dataframe based on user selection
        filtered_df = agri_prod_seasonal[agri_prod_seasonal['Crop'] == selected_crop]

        # Create the stacked bar chart for production trends with seasonal breakdown
        fig1 = px.bar(filtered_df,
                      x='Year',
                      y='Value',
                      color='Season',
                      labels={'Value': 'Production'},
                      title=f'Production Trend for {selected_crop}',
                      height=600)

        # Update layout for better readability
        fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # Display the chart
        st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Bar plot for Year-on-Year percentage changes of the selected crop for the last 10 years using agri_prod_totals
    with plot_col2:
        # Filter data for the selected crop using agri_prod_totals
        selected_crop_totals_df = agri_prod_totals[agri_prod_totals['Crop'] == selected_crop]

        # Calculate the Year-on-Year percentage change
        selected_crop_totals_df = selected_crop_totals_df.sort_values(by='Year').reset_index(drop=True)
        selected_crop_totals_df['YoY_Change'] = selected_crop_totals_df['Value'].pct_change() * 100

        # Filter to only include the last 10 years of data
        selected_crop_totals_df_last_10_years = selected_crop_totals_df.tail(10)

        # Create a bar plot for Year-on-Year percentage changes for the last 10 years
        fig2 = px.bar(
            selected_crop_totals_df_last_10_years,
            x='Year',
            y='YoY_Change',
            text=selected_crop_totals_df_last_10_years['YoY_Change'].round(1),
            labels={'YoY_Change': 'Y-o-Y (%)'},
            title=f'Y-o-Y (%) Change in Production for {selected_crop} (Last 10 Years)',
            height=600,
            color_discrete_sequence=['#1f77b4']
        )

        # Update the bar plot's layout for better readability
        fig2.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title='Y-o-Y % Change',
            xaxis_title='Year'
        )
        fig2.update_traces(textposition='outside')

        # Display the chart
        st.plotly_chart(fig2, use_container_width=True)

    # Optional: Display the filtered dataframe
    if st.checkbox('Show data'):
        st.write(filtered_df)
        
    st.header("Horticultural Production | Yearly")
    
    # Display production metrics
    col1, col2, col3 = st.columns(3)
    citrus_latest, citrus_change = get_latest_and_change(horti_long, 'Citrus Total (i to iv)')
    fruits_latest, fruits_change = get_latest_and_change(horti_long, 'Total Fruits')
    veg_latest, veg_change = get_latest_and_change(horti_long, 'Total Vegetables')

    #col1.metric("Citrus Production", f"{citrus_latest:.2f} MT", f"{citrus_change:.2f}")
    col1.metric("Fruits Production", f"{fruits_latest/100000:.2f} lakh MT", f"{fruits_change:.2f}")
    col2.metric("Vegetables Production", f"{veg_latest/10000:.2f} lakh MT", f"{veg_change:.2f}")

    # Create two columns for plots
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        selected_crop = st.selectbox('Select a crop', options=sorted(horti_long['Crops'].unique()))
        filtered_df = horti_long[horti_long['Crops'] == selected_crop].sort_values(by='Year')

        fig1 = px.bar(filtered_df, x='Year', y='Production_in_tonnes', title=f'Production Trend for {selected_crop}')
        fig1.add_scatter(x=filtered_df['Year'], y=filtered_df['Area'], mode='lines+markers', name='Area', yaxis='y2')
        fig1.update_layout(yaxis2=dict(title='Area (hectares)', overlaying='y', side='right', showgrid=False))
        st.plotly_chart(fig1, use_container_width=True)

    with plot_col2:
        filtered_df['YoY_Change'] = filtered_df['Production_in_tonnes'].pct_change() * 100
        fig2 = px.bar(filtered_df.tail(10), x='Year', y='YoY_Change', 
                      text=filtered_df.tail(10)['YoY_Change'].round(1),
                      title=f'Y-o-Y Change in Production for {selected_crop}',
                      color_discrete_sequence=['#1f77b4'])
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

with tab4:
    commodity_dict = {
    "Gram": 1, "Groundnut": 3, "Masur/Lentil (Dal/Split)": 5, "Groundnut Oil": 4, "Lentil": 6, "Moong (Dal/Split)": 7, "Moong": 8, "Onion": 9,
    "Paddy": 10, "Potato": 12, "Rapeseed & Mustard Oil": 14, "Rapeseed & Mustard": 13, "Rice": 15, "Soybean": 16, "Sunflower": 18,
    "Tur (Dal/Split)": 21, "Tomato": 20, "Tur": 22, "Urad (Dal/Split)": 23, "Urad": 24, "Wheat Atta": 26, "Wheat": 25, "Sesamum": 27,
    "Avare Dal": 32, "Bajra": 33, "Barley": 34, "Castorseed": 36, "Cotton": 37, "Foxtail Millet": 41, "Cowpea": 40, "Jute": 45,
    "Guarseed": 43, "Jowar": 44, "Kulthi": 48, "Kodo Millet": 47, "Lakh": 49, "Linseed": 50, "Maize": 53, "Nigerseed": 56, "Peas": 58,
    "Ragi": 62, "Ramdana": 64, "Safflower": 66, "Sannhemp": 69, "Sugarcane": 72, "Tobacco": 74}
    
    # Create dropdown menu
    selected_commodity = st.selectbox(
    'Select a commodity',
    options=sorted(commodity_dict.keys()),  # Sort alphabetically
    index = 23  # Default to first item
    )

    # Get the ID of selected commodity
    selected_commodity_id = commodity_dict[selected_commodity]
    
    data_object = fetch_and_process_data(
        commodity_id=selected_commodity_id,
        month_from=1,
        year_from=2014,
        month_to=12,
        year_to=dt.datetime.today().year
    )
    data_arr = data_object.dataframe
    title_arrivals = data_object.title
    
    plot_comparison(data_arr)
