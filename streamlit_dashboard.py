import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json
import requests
from bs4 import BeautifulSoup
import re
from functools import lru_cache

def wide_space_default():
    st.set_page_config(layout='wide')

wide_space_default()

# Load and preprocess DCA data (Cached function)
@st.cache_data
def load_dca_data():
    df = pd.read_excel('dca_data.xlsx', sheet_name="State_Consolidated_TimeSeries").iloc[54:76, 1:].T.reset_index()
    cols = df.iloc[0, 1:]
    index = df.iloc[1:, 0]
    df = df.iloc[1:, 1:]
    df.columns = cols
    df.index = index
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
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

# Function to get latest production and change in value for a specific crop
def get_latest_and_change(df, crop, metric='Production_in_tonnes'):
    crop_data = df[df['Crops'] == crop].sort_values('Year')
    latest = crop_data[metric].iloc[-1]
    previous = crop_data[metric].iloc[-2]
    change = 100 * (latest - previous) / previous
    return latest, change

# Load the GeoJSON file for Indian states
@st.cache_data
def load_geojson():
    with open('india_states.geojson', 'r') as f:
        return json.load(f)

# Function to fetch and process rainfall data
@st.cache_data
def fetch_rainfall_data(rainfall_type):
    url = f"https://mausam.imd.gov.in/responsive/rainfallinformation_state.php?msg={rainfall_type}"
    response = requests.get(url)
    html = response.text

    soup = BeautifulSoup(html, 'html.parser')
    script_tag = soup.find('script', text=lambda t: t and 'var mapVar = AmCharts.parseGeoJSON' in t)
    data_start = script_tag.string.index('"areas": [')
    data_end = script_tag.string.index(']', data_start) + 1
    json_data = script_tag.string[data_start:data_end]

    json_data = json_data.replace('"areas": ', '')
    json_data = re.sub(r'(\w+):', r'"\1":', json_data)
    areas_data = json.loads(json_data)

    rainfall_data = []
    for area in areas_data:
        if area['id'] and area['id'] != 'null':
            state = area['title'].strip()
            balloon_data = extract_data(area['balloonText'])
            rainfall_data.append({
                'state': state,
                'actual': balloon_data['actual'],
                'normal': balloon_data['normal'],
                'deviation': balloon_data['deviation']
            })

    df = pd.DataFrame(rainfall_data)
    df['state'] = df['state'].apply(lambda x: x.title().replace(" (Ut)", "").replace("&", "and") if "Jammu" in x else x.title().replace(" (Ut)", ""))
    
    return df

# Helper function to extract data from the balloon text
def extract_data(balloon_text):
    actual = re.search(r'Actual : ([\d.]+) mm', balloon_text)
    normal = re.search(r'Normal : ([\d.]+) mm', balloon_text)
    departure = re.search(r'Departure : ([-\d]+)%', balloon_text)

    return {
        'actual': float(actual.group(1)) if actual else None,
        'normal': float(normal.group(1)) if normal else None,
        'deviation': int(departure.group(1)) if departure else None
    }

# Function to calculate group metrics for production data
def calculate_group_metrics(group_name, agri_prod_totals):
    latest_year = agri_prod_totals['Year'].max()
    previous_year = latest_year - 1
    latest_value = agri_prod_totals[(agri_prod_totals['Crop'].str.contains(group_name)) & 
                                    (agri_prod_totals['Year'] == latest_year)]['Value'].sum()
    previous_value = agri_prod_totals[(agri_prod_totals['Crop'].str.contains(group_name)) & 
                                      (agri_prod_totals['Year'] == previous_year)]['Value'].sum()
    delta = ((latest_value - previous_value) / previous_value) * 100 if previous_value != 0 else 0
    return latest_value, delta

# Load data
df_long = load_dca_data()
agri_prod_long = load_production_data()
horti_long = load_horticulture_data()
india_geojson = load_geojson()

# Get the min and max dates for the slider
min_date = df_long['Date'].min()
max_date = df_long['Date'].max()
three_months_ago = max_date - pd.DateOffset(months=3)
df_long['Days'] = (df_long['Date'] - min_date).dt.days

# Filter the data for the last three months by default
df_long_default = df_long[df_long['Date'] >= three_months_ago]

# Streamlit app
st.title("Inflation Monitoring Dashboard")

tab1, tab2, tab3 = st.tabs(["DCA Retail Price Trends", "Rainfall Deviation", "Agricultural Production Trends"])

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


