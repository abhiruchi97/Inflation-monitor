# Load and preprocess DCA data (Cached function)
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json
import requests
from bs4 import BeautifulSoup
import re
from functools import lru_cache
from utils.helper import *
from typing import Tuple, Optional
from dataclasses import dataclass
from typing import Dict, List

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

@st.cache_data
def load_cpi_data():
    data = pd.read_excel("inflation_2012-24.xlsx", index_col = 'Description').iloc[[12,18,26,27],3:]
    return data

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

# Functions for showing metrics of CPI categories
def get_broad_metrics(data, category):
    
    latest = data.loc[category][-1].round(1)
    previous = data.loc[category][-2].round(1)
    
    latest_month = data.loc[category].index[-1].strftime("%b %Y")
    
    return {"current mom": latest, 
            "previous mom": previous,
            "month": latest_month}

def custom_metric(label, current, previous):
    st.markdown(f"""
        <style>
            .metric-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 1rem; background: #2d2d2d; border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2); width: 300px; margin: 1rem;
            }}
            .metric-label {{ font-size: 0.875rem; color: #9ca3af; margin-bottom: 0.5rem; }}
            .metric-value {{ font-size: 1.875rem; font-weight: 600; margin-bottom: 0.25rem; color: #ffffff; }}
            .metric-delta {{ font-size: 0.875rem; color: #9ca3af; }}
            .lower {{ color: #4ade80 !important; }}
            .higher {{ color: #f87171 !important; }}
        </style>
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value {'lower' if current < previous else 'higher'}">{current}</div>
            <div class="metric-delta">Previous: {previous}</div>
        </div>
    """, unsafe_allow_html=True)

# Arrival data - Wholesale prices and Arrivals
# Function to fetch data from UPAG through API calls and generating sorted dataframe

def fetch_price_data(commodity_id, month_from=None, year_from=None, month_to=None, year_to=None):
    url = "https://dash.upag.gov.in/_dash-update-component"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "Content-Type": "application/json",
        "Origin": "https://dash.upag.gov.in",
        "Referer": "https://dash.upag.gov.in/pricesmonthcomparison?t=&stateID=0&rtab=Analytics&rtype=dashboards",
    }
    
    # Base payload structure
    payload = {
        "output": "..prices-graph-mixed.figure...prices-graph-source.children..",
        "outputs": [
            {"id": "prices-graph-mixed", "property": "figure"},
            {"id": "prices-graph-source", "property": "children"}
        ],
        "inputs": [
            {
                "id": "prices-graph-filters-store",
                "property": "data",
                "value": {
                    "monthfrom": month_from,
                    "yearfrom": year_from,
                    "monthto": month_to,
                    "yearto": year_to,
                    "commodity": commodity_id,
                    "source": [10, 13]
                }
            },
            {
                "id": "prices-graph-location-filter-aggrid",
                "property": "selectedRows",
                "value": [{"State": "All India", "StateCode": 999999}]
            },
            {
                "id": "prices-body-tab",
                "property": "value",
                "value": "graph"
            }
        ],
        "changedPropIds": ["prices-graph-filters-store.data"],
        "state": [
            {
                "id": "url",
                "property": "search",
                "value": "?t=&stateID=0&rtab=Analytics&rtype=dashboards"
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, verify = False)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

@dataclass
class CommodityResponse:
    """Data class to hold processed commodity data and title"""
    dataframe: pd.DataFrame
    title: str

def extract_figure_data(json_data: Dict) -> Tuple[List[Dict], str]:
    """Extract figure data and title from JSON response"""
    try:
        figure_data = json_data["response"]["prices-graph-mixed"]["figure"]
        return (
            figure_data["data"],
            figure_data["layout"]["title"]["text"]
        )
    except KeyError as e:
        raise ValueError(f"Invalid JSON structure: missing key {e}")

def create_month_mapping() -> Dict[str, int]:
    """Create a mapping of month names to numbers"""
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    return {month: idx + 1 for idx, month in enumerate(months)}

def process_commodity_data(json_data: Dict) -> CommodityResponse:
    """
    Process commodity data from JSON response and return structured data.
    
    Args:
        json_data: Raw JSON response from the API
        
    Returns:
        CommodityResponse object containing processed DataFrame and title
    """
    try:
        # Extract graph data and title
        graph_data, title = extract_figure_data(json_data)
        
        # Initialize records list
        records = []
        
        # Process each data series
        for series in graph_data:
            # Extract year and data type
            name_parts = series["name"].split()
            if len(name_parts) < 2:
                continue
                
            year = int(name_parts[1])
            data_type = "Price" if "Prices" in series["name"] else "Arrivals"
            
            # Create records for each month-value pair
            for month, value in zip(series["x"], series["y"]):
                key = (year, month)
                
                # Find or create record
                record = next(
                    (r for r in records if (r["Year"], r["Month"]) == key),
                    {"Year": year, "Month": month}
                )
                
                if record not in records:
                    records.append(record)
                    
                record[data_type] = value

        # Convert to DataFrame
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("No data extracted from JSON")

        # Sort by year and month
        month_mapping = create_month_mapping()
        df_sorted = (df
                    .assign(Month_num=df['Month'].map(month_mapping))
                    .sort_values(['Year', 'Month_num'])
                    .drop('Month_num', axis=1)
                    .reset_index(drop=True))
        
        return CommodityResponse(
            dataframe=df_sorted,
            title=title
        )
        
    except Exception as e:
        raise ValueError(f"Error processing commodity data: {str(e)}")

## Main function that calls on functions above to generate dataframe
def fetch_and_process_data(commodity_id: int, **kwargs) -> CommodityResponse:
    """
    Fetch and process commodity data in one step.
    
    Args:
        commodity_id: ID of the commodity to fetch
        **kwargs: Additional arguments for fetch_price_data
        
    Returns:
        CommodityResponse object containing dataframe and title with commodity name.
    
    Example usage:
    result = fetch_and_process_data(
        commodity_id=12,
        month_from=1,
        year_from=2014,
        month_to=12,
        year_to=2024
    )

    df = result.dataframe
    title = result.title
    """
    # Using the fetch_price_data function from your previous code
    json_data = fetch_price_data(commodity_id, **kwargs)
    if not json_data:
        raise ValueError("Failed to fetch data from API")
        
    return process_commodity_data(json_data)
        
# Generate plots for wholesale prices and arrivals
def plot_comparison(df):
    """
    Plots grouped bar comparisons with 2014-2019 average line overlay.
    """
    # Create month ordering
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Get the latest 3 years for bars
    latest_years = sorted(df['Year'].unique())[-3:]
    
    # Calculate 2014-2019 average
    historical_avg = df[df['Year'].between(2014, 2019)].groupby('Month').agg({
        'Arrivals': 'mean',
        'Price': 'mean'
    }).reset_index()
    historical_avg['Month'] = pd.Categorical(historical_avg['Month'], categories=month_order, ordered=True)
    historical_avg = historical_avg.sort_values('Month')
    
    # Filter and prepare data for bars
    df_filtered = df[df['Year'].isin(latest_years)].copy()
    df_filtered['Month'] = pd.Categorical(df_filtered['Month'], categories=month_order, ordered=True)
    df_filtered = df_filtered.sort_values(['Month', 'Year']).reset_index(drop=True)
    df_filtered['Year'] = df_filtered['Year'].astype(str)
    
    # Create arrival comparison plot
    fig_arrivals = px.bar(
        df_filtered,
        x='Month',
        y='Arrivals',
        color='Year',
        title=f'Monthly Arrivals: ({latest_years[0]}-{latest_years[-1]} and 2014-2019 Average)',
        labels={'Arrivals': 'Arrivals (MT)', 'Year': 'Year'},
        category_orders={
            'Month': month_order,
            'Year': [str(year) for year in latest_years]
        },
        barmode='group',
        #color_discrete_sequence=['#1f77b4', '#2ca02c', '#ff7f0e']
    )
    
    # Add historical average line
    fig_arrivals.add_scatter(
        x=historical_avg['Month'],
        y=historical_avg['Arrivals'],
        mode='lines+markers',
        line=dict(color='purple', dash='dash'),
        name='Average 2014-2019',
        marker=dict(symbol='diamond')
    )
    
    # Update arrivals layout
    fig_arrivals.update_layout(
        height=500,
        showlegend=True,
        xaxis_title='Month',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        bargap=0.15,
        bargroupgap=0.1
    )
    
    # Create price comparison plot
    fig_prices = px.bar(
        df_filtered,
        x='Month',
        y='Price',
        color='Year',
        title=f'Average Wholesale Prices: ({latest_years[0]}-{latest_years[-1]} and 2014-2019 Average)',
        labels={'Price': 'Price (Rs/Quintal)', 'Year': 'Year'},
        category_orders={
            'Month': month_order,
            'Year': [str(year) for year in latest_years]
        },
        barmode='group',
        #color_discrete_sequence=['#1f77b4', '#2ca02c', '#ff7f0e']
    )
    
    # Add historical average line
    fig_prices.add_scatter(
        x=historical_avg['Month'],
        y=historical_avg['Price'],
        mode='lines+markers',
        line=dict(color='purple', dash='dash'),
        name='Average 2014-2019',
        marker=dict(symbol='diamond')
    )
    
    # Update prices layout
    fig_prices.update_layout(
        height=500,
        showlegend=True,
        xaxis_title='Month',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        bargap=0.15,
        bargroupgap=0.1
    )
    
    # Display plots in Streamlit columns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_arrivals, use_container_width=True)
    with col2:
        st.plotly_chart(fig_prices, use_container_width=True)