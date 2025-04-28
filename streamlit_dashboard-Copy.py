import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go # Added for scatter trace in daily arrivals
import streamlit as st
import json
import datetime # Use datetime directly
import requests
from bs4 import BeautifulSoup
import re
from functools import lru_cache
from utils.helper import * # Assuming this helper module exists and is needed for other tabs
from typing import Tuple, Optional
from dataclasses import dataclass
from typing import Dict, List
# Imports needed for the new Global Inflation tab
# import seaborn as sns
# import matplotlib.pyplot as plt
import urllib3 # Already present, but confirm needed


# Suppress SSL warnings if needed
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # Already present

def wide_space_default():
    st.set_page_config(layout='wide')

wide_space_default()

# --- Functions for Existing Tabs ---
@st.cache_data
def load_daily_arrival_data():
        # Make sure the path is correct
        try:
            return pd.read_csv("commodity_data.csv")
        except FileNotFoundError:
            st.error("Error: commodity_data.csv not found.")
            return None

@st.cache_data
def load_dca_data():
    try:
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
    except FileNotFoundError:
        st.error("Error: dca_test.xlsx not found.")
        return None

@st.cache_data
def load_cpi_data():
    data = pd.read_excel("inflation_2012-24.xlsx", index_col = 'Description').iloc[[12,18,26,27],3:]
    return data

# Load and preprocess production data (Cached function)
@st.cache_data
def load_production_data():
    try:
        agri_prod = pd.read_excel('dca_data_1.xlsx', sheet_name='test')
        agri_prod['Crop'] = agri_prod['Crop'].fillna(method='ffill')

        # Convert the Year columns to a consistent format
        agri_prod.columns = [str(int(col.split('-')[0]) + 1) if isinstance(col, str) and '-' in col else col for col in agri_prod.columns]

        # Convert to long format, including the totals
        agri_prod_long = pd.melt(agri_prod,
                                 id_vars=['Crop', 'Season'],
                                 var_name='Year',
                                 value_name='Value')

        # Convert Year column to numeric
        agri_prod_long['Year'] = pd.to_numeric(agri_prod_long['Year'].apply(lambda x: str(x)[:2] + str(x)[-2:] if isinstance(x, str) else x))


        return agri_prod_long
    except FileNotFoundError:
        st.error("Error: dca_data_1.xlsx not found.")
        return None


# Load and preprocess horticulture data
@st.cache_data
def load_horticulture_data():
    try:
        horti_df = pd.read_excel('dca_data_1.xlsx', sheet_name='horti', header=[0, 1], index_col=[0])
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
    except FileNotFoundError:
        st.error("Error: dca_data_1.xlsx not found.")
        return None

# Load the GeoJSON file for Indian states
@st.cache_data
def load_geojson():
    try:
        with open('india_states.geojson', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: india_states.geojson not found.")
        return None

# --- Helper Functions for Global Inflation Tab ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_inflation_data(file_path="inflation_long.csv"):
    """Loads and preprocesses the global inflation data."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Error: Global inflation data file not found at {file_path}")
        return None

@st.cache_data(ttl=3600)
def pivot_inflation_data(_df):
    """Pivots the global inflation dataframe."""
    if _df is None:
        return None
    df_pivot_full = _df.pivot(index='Country', columns='Date', values='Inflation')
    df_pivot_full = df_pivot_full.sort_index(axis=1) # Sort columns by date
    return df_pivot_full

def normalize_row_inflation(row):
    """Applies min-max normalization to a pandas Series (row for inflation)."""
    min_val = row.min()
    max_val = row.max()
    denominator = max_val - min_val
    if pd.isna(denominator) or denominator == 0:
        return pd.Series(0.5 if denominator == 0 else np.nan, index=row.index)
    else:
        return (row - min_val) / denominator

@st.cache_data(ttl=3600)
def normalize_inflation_full_history(_df_pivot_full):
    """Normalizes the pivoted inflation data row-wise based on full history."""
    if _df_pivot_full is None:
        return None
    return _df_pivot_full.apply(normalize_row_inflation, axis=1)

@st.cache_data(ttl=3600) # Cache data loading for 1 hour
def load_spf_data(file_path="latest_spf_data.csv"):
    """Loads the processed SPF data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Attempt to convert relevant columns to datetime, ignore errors if already correct type
        for col in ['last_update_time', 'forecast_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        # Ensure 'value' is numeric
        if 'value' in df.columns:
             df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value']) # Drop rows where value conversion failed
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data from {file_path}: {e}")
        return None
    

# # --- Updated Plotting Function using Plotly Express ---
def create_inflation_heatmap(df_normalized_subset, df_pivot_subset, start_date_str, end_date_str):
    """Generates the inflation heatmap plot using Plotly Express."""
    if df_normalized_subset is None or df_normalized_subset.empty:
        st.warning(f"No global inflation data available for the selected period: {start_date_str} to {end_date_str}.")
        return None

    # --- Data Preparation ---
    try:
        # Get formatted dates for x-axis labels
        x_labels = df_normalized_subset.columns.strftime('%b-%Y')
        # Get country names for y-axis labels
        y_labels = df_normalized_subset.index.tolist()
        # Use normalized values for coloring
        color_values = df_normalized_subset.values
        # Get original values for text annotations
        text_values_df = df_pivot_subset # Keep as DataFrame for easier iteration
    except Exception as e:
        st.error(f"Error preparing data for Plotly Express heatmap: {e}")
        return None

    # --- Create Plotly Express Heatmap ---
    # Use normalized data for color mapping (zmin/zmax)
    fig = px.imshow(
        df_normalized_subset,       # Input data for structure and color mapping
        x=x_labels,                 # Set x-axis labels explicitly
        y=y_labels,                 # Set y-axis labels explicitly
        color_continuous_scale='RdYlGn_r', # Red=High, Green=Low (reversed scale)
        zmin=0,                     # Map colors based on 0-1 normalized range
        zmax=1,
        aspect="auto",              # Allow non-square cells based on figure size
        text_auto=False             # Disable automatic text annotations from input data
    )

    # --- Customize Hover Information ---
    fig.update_traces(
        customdata=text_values_df.values, # Pass original values as customdata
        hovertemplate="<b>Country:</b> %{y}<br>" +
                      "<b>Date:</b> %{x}<br>" +
                      "<b>Inflation:</b> %{customdata:.1f}%" + # Access original value from customdata
                      "<extra></extra>" # Removes the trace name info
    )

    # --- Manually Add Annotations from Original Data ---
    annotations = []
    for r_idx, country in enumerate(y_labels):
        for c_idx, date_label in enumerate(x_labels):
            original_value = text_values_df.iloc[r_idx, c_idx]
            if pd.notna(original_value): # Only add annotation if value exists
                annotations.append(
                    go.layout.Annotation(
                        text=f"{original_value:.1f}", # Format original value
                        x=date_label,          # Position based on date label
                        y=country,             # Position based on country label
                        xref='x',
                        yref='y',
                        showarrow=False,
                        font=dict(size=10, color="black") # Adjust font size/color
                    )
                )

    # --- Update Layout ---
    fig.update_layout(
        title=dict( # Use dict for more title options
            text=f'Global Headline Inflation Heatmap',
            x=0.5, # Center the title
            xanchor='center' # Anchor title centrally
        ),        
        xaxis_side='bottom',             # Move dates (x-axis) to the top
        xaxis_tickangle=-90,          # Rotate x-axis labels
        yaxis_autorange='reversed',   # Display countries top-to-bottom
        # --- Control Size ---
        width=1200,                    # Set desired width (pixels)
        height=700,                   # Set desired height (pixels)
        margin=dict(l=50, r=50, t=50, b=50), # Adjust margins
        annotations=annotations       # Add the manual annotations
    )

    # --- Customize Color Bar ---
    fig.update_coloraxes(colorbar=dict(
            title='',
            titleside='right',
            tickvals=[0.05, 0.95],     # Positions relative to 0-1
            ticktext=['Low', 'High'],
            lenmode='fraction',
            len=0.75,
            thickness=15
        )
    )


    return fig

def get_fy_quarter_label(date_obj):
    """
    Calculates the financial quarter (Q1-Q4) and year (YYYY-YY)
    for a given date, assuming an April-March financial year.
    """
    if not isinstance(date_obj, (datetime.date, pd.Timestamp)):
         # Handle cases where date_obj might be NaT or None after conversion errors
         return None, None

    month = date_obj.month
    year = date_obj.year
    if 1 <= month <= 3: # Jan-Mar -> Q4 of previous FY
        quarter = 4
        fy_start = year - 1
    elif 4 <= month <= 6: # Apr-Jun -> Q1
        quarter = 1
        fy_start = year
    elif 7 <= month <= 9: # Jul-Sep -> Q2
        quarter = 2
        fy_start = year
    elif 10 <= month <= 12: # Oct-Dec -> Q3
        quarter = 3
        fy_start = year
    else: # Should not happen with valid date object
        return None, None

    fy_end_short = (fy_start + 1) % 100
    fy_label = f"{fy_start}-{fy_end_short:02d}"
    return quarter, fy_label

def map_period_to_quarter_label(original_period, reference_date):
    """
    Maps generic period labels ('Current Quarter', 'Next Quarter', etc.)
    to specific financial quarter labels ('QX:YYYY-YY') based on a reference date.
    """
    if reference_date is None:
        return original_period # Cannot map without reference date

    # Get the month *before* the reference date
    try:
        # Ensure reference_date is a suitable type for offset calculation
        if isinstance(reference_date, datetime.date) and not isinstance(reference_date, datetime.datetime):
             reference_date_ts = pd.Timestamp(reference_date)
        else:
             reference_date_ts = reference_date

        prev_month_date = reference_date_ts - pd.DateOffset(months=1)
    except Exception as e:
         # Handle potential errors if reference_date is not date-like
         # print(f"Debug: Error calculating previous month from {reference_date}: {e}")
         return original_period


    # Get the starting quarter and FY label based on the previous month
    start_quarter, start_fy = get_fy_quarter_label(prev_month_date)
    if start_quarter is None: # Handle error from get_fy_quarter_label
         return original_period

    # Determine quarter offset based on the original period label
    offset_map = {
        "Current Quarter": 0,
        "Next Quarter": 1,
        "Next 2 Quarters": 2,
        "Next 3 Quarters": 3,
        "Next 4 Quarters": 4
    }
    offset = offset_map.get(original_period)
    if offset is None:
        return original_period # Return original if not a recognized period label

    # Calculate target quarter number (1-based) and year adjustments
    current_quarter_num_zero_based = start_quarter - 1 # Convert start quarter to 0-based index (0-3)
    target_quarter_num_zero_based = current_quarter_num_zero_based + offset
    year_offset = target_quarter_num_zero_based // 4 # How many full years to advance
    final_quarter = (target_quarter_num_zero_based % 4) + 1 # Target quarter number (1-4)

    # Calculate target financial year start based on the initial FY start year
    try:
        start_fy_year = int(start_fy.split('-')[0])
        target_fy_start = start_fy_year + year_offset
        target_fy_end_short = (target_fy_start + 1) % 100
        target_fy_label = f"{target_fy_start}-{target_fy_end_short:02d}"
    except Exception as e:
         # Handle potential errors parsing start_fy
         # print(f"Debug: Error calculating target FY label from {start_fy}: {e}")
         return original_period


    return f"Q{final_quarter}:{target_fy_label}"

def render_spf_expectations_tab(csv_path="latest_spf_data.csv"):
    """
    Renders the SPF Inflation Expectations tab content within a Streamlit app.

    Args:
        csv_path (str): The path to the processed SPF data CSV file.
                        Assumes columns: 'value', 'last_update_time', 'forecast_date',
                        'type', 'indicator', 'period'.
    """

    st.header("Survey of Professional Forecasters")
    st.subheader("CPI Inflation Expectations")

    # --- Load Data ---
    df = load_spf_data(csv_path)

    if df is None:
        st.warning("Could not load SPF data. Please ensure 'latest_spf_data.csv' exists and is correctly formatted.")
        return # Stop execution for this tab if data loading fails

    # --- Optional Raw Data Display ---
    if st.checkbox("Show Loaded SPF Data", key="show_spf_raw"):
        st.dataframe(df)

    # --- Select Plot Type ---
    selected_type = st.radio(
        "Select Forecast Horizon:",
        ('Quarterly', 'Annual'),
        key="spf_type_select",
        horizontal=True # Display options side-by-side
    )

    # --- Filter Data Based on Selection ---
    df_filtered = df[df['type'] == selected_type].copy()

    # --- Get Common Update Time (Reference Date) ---
    common_last_update_time = None
    if not df_filtered.empty and 'last_update_time' in df_filtered.columns:
         valid_update_times = df_filtered['last_update_time'].dropna()
         if not valid_update_times.empty:
              common_last_update_time = valid_update_times.max() # Use the latest date as reference


    # --- Prepare and Plot Based on Selection ---
    fig = None # Initialize fig to None
    if selected_type == 'Quarterly':
        # --- Prepare Quarterly Data ---
        period_order = [
            "Current Quarter", "Next Quarter", "Next 2 Quarters",
            "Next 3 Quarters"#, "Next 4 Quarters"
        ]
        # Filter out periods not in the defined order first
        df_filtered = df_filtered[df_filtered['period'].isin(period_order)].copy() # Use copy after filtering

        if not df_filtered.empty:
            # Apply the mapping to get formatted labels
            if common_last_update_time:
                 df_filtered['formatted_period'] = df_filtered['period'].apply(
                     lambda p: map_period_to_quarter_label(p, common_last_update_time)
                 )
            else:
                 # Fallback if no reference date is found
                 st.warning("Could not determine the reference date ('last_update_time'). Using generic period labels.")
                 df_filtered['formatted_period'] = df_filtered['period']

            # Use original 'period' for logical sorting via categorical type
            df_filtered['period'] = pd.Categorical(
                df_filtered['period'], categories=period_order, ordered=True
            )
            df_filtered = df_filtered.sort_values(by=['indicator', 'period'])

            # --- Plot Quarterly Data ---
            plot_title = "Quarterly Inflation Expectations Trajectory"
            if common_last_update_time:
                plot_title += f" (as of policy dated {common_last_update_time.strftime('%B %d, %Y')})"

            fig = px.line(
                df_filtered,
                x='formatted_period', # Use the new formatted labels for the x-axis display
                y='value',
                color='indicator',
                markers=True,
                text='value',
                labels={
                    'formatted_period': 'Forecast Period', # Update label
                    'value': 'Median Inflation Expectation (Y-o-Y)',
                    'indicator': 'Indicator Type'
                },
                title=plot_title
            )
            fig.update_layout(
                legend_title_text='Indicator',
                # Ensure x-axis ticks match the data points even if categorical mapping isn't perfect
                xaxis=dict(type='category'), # Treat x-axis as categorical based on the formatted labels
                yaxis=dict(ticksuffix='%')
            )

                # Adjust how the text labels appear
            fig.update_traces(
            textposition="top center",      # <-- Position the labels nicely
            texttemplate='%{text:.1f}'     # <-- Format: 2 decimal points + '%' sign
            )
            # Optional: Explicitly set tick labels if needed, though usually Plotly handles it
            # fig.update_xaxes(tickvals=df_filtered['formatted_period'].unique(), ticktext=df_filtered['formatted_period'].unique())


    elif selected_type == 'Annual':
        # --- Prepare Annual Data ---
        period_order_annual = [
            "Current Fiscal Year",
            "Next Fiscal Year"
        ]
        # Filter out periods not in the defined order
        df_filtered = df_filtered[df_filtered['period'].isin(period_order_annual)].copy() # Use copy

        if not df_filtered.empty:
            df_filtered['period'] = pd.Categorical(
                df_filtered['period'], categories=period_order_annual, ordered=True
            )
            df_filtered = df_filtered.sort_values(by=['indicator', 'period'])

            # --- Plot Annual Data ---
            plot_title = "Annual Inflation Expectations"
            if common_last_update_time:
                plot_title += f" (as of policy dated {common_last_update_time.strftime('%B %d, %Y')})"

            fig = px.bar(
                df_filtered, x='period', y='value', color='indicator',
                barmode='group',
                labels={
                    'period': 'Fiscal Year',
                    'value': 'Median Inflation Expectation (Y-o-Y)',
                    'indicator': 'Indicator Type'
                },
                title=plot_title,
                text_auto='.1f'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                 legend_title_text='Indicator',
                 yaxis=dict(ticksuffix='%')
            )

    # --- Display Plot ---
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        # Optionally display the data table used for the plot
        if st.checkbox(f"Show Filtered {selected_type} Data Table", key=f"show_spf_{selected_type.lower()}_table"):
             display_cols = ['indicator', 'period', 'formatted_period' if selected_type == 'Quarterly' else 'period', 'value', 'forecast_date', 'last_update_time']
             display_cols = [col for col in display_cols if col in df_filtered.columns] # Ensure columns exist
             st.dataframe(df_filtered[display_cols])

    elif not df.empty and df_filtered.empty:
         # This means data was loaded, but filtering resulted in empty dataframe
         st.info(f"No {selected_type.lower()} data found matching the expected period labels (e.g., 'Current Quarter', 'Next Fiscal Year') in the loaded file.")
    elif df.empty:
         # This case handled by the initial load check, but added for completeness
         pass
    else:
         # Generic case if fig is None but df_filtered wasn't technically empty (e.g., error during plotting)
         st.info(f"Could not generate plot for {selected_type.lower()} data.")


# --- Main App Starts Here ---

# Load data for existing tabs (handle potential None return values)
df_long = load_dca_data()
agri_prod_long = load_production_data()
horti_long = load_horticulture_data()
india_geojson = load_geojson()
# Assume cpi_data and helper functions for CPI metrics are loaded/defined elsewhere or in utils.helper
cpi_data = load_cpi_data() # Make sure this function exists or data is loaded

# Check if essential data loaded, otherwise stop or show error
if df_long is None or agri_prod_long is None or horti_long is None or india_geojson is None:
    st.error("Essential data files could not be loaded. Dashboard cannot proceed.")
    st.stop() # Stop execution if core data is missing

# Calculate date ranges and default values if df_long loaded correctly
min_date = df_long['Date'].min()
max_date = df_long['Date'].max()
three_months_ago = max_date - pd.DateOffset(months=3)
df_long['Days'] = (df_long['Date'] - min_date).dt.days
df_long_default = df_long[df_long['Date'] >= three_months_ago]

# List of crops (ensure this list is comprehensive if used across tabs)
list_of_crops = ['Rice', 'Wheat', 'Maize', 'Barley', 'Jowar', 'Bajra', 'Ragi', 'Small Millets',
        'Shree Anna /Nutri Cereals', 'Nutri/Coarse Cereals', 'Cereals', 'Tur', 'Gram',
        'Urad', 'Moong', 'Lentil', 'Other Pulses', 'Total Pulses', 'Total Food Grains',
        'Groundnut', 'Castorseed', 'Sesamum', 'Nigerseed', 'Soybean', 'Sunflower',
        'Rapeseed & Mustard', 'Linseed', 'Safflower', 'Total Oil Seeds', 'Sugarcane',
        'Cotton', 'Jute', 'Mesta', 'Jute & Mesta', 'Tobacco', 'Sannhemp', 'Guarseed']


# Streamlit app layout
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

# --- Define Tabs ---
tab1, tab2, tab3, tab4, tab_global_inflation, tab_spf_plot = st.tabs([
    "DCA Retail Price Trends",
    "Rainfall Deviation",
    "Food Production Trends",
    "Daily Mandi Arrivals",
    "Global Inflation",
    "SPF Forecasts"
])

# --- Tab 1: DCA Retail Price Trends ---
with tab1:
    st.header("DCA Retail Price Trends")
    url_dca = "https://fcainfoweb.nic.in/reports/report_menu_web.aspx"

    col1, col2 = st.columns(2)

    with col1:
        all_commodities_option = ["All Commodities"] + list(df_long['Commodity'].unique())
        commodities_selected = st.multiselect(
            "Select commodities",
            options=all_commodities_option,
            default=['Tomato', 'Potato', 'Onion']
        )

        if "All Commodities" in commodities_selected:
            commodities = list(df_long['Commodity'].unique())
        else:
            commodities = commodities_selected

        normalize = st.checkbox("Normalize prices to 100")
        start_date_dca = st.date_input("Start date", value=three_months_ago.date(), min_value=min_date.date(), max_value=max_date.date(), key="dca_start")
        end_date_dca = st.date_input("End date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date(), key="dca_end")

        # Use the correct date variables for filtering
        filtered_df_long = df_long[(df_long['Commodity'].isin(commodities)) &
                                (df_long['Date'] >= pd.Timestamp(start_date_dca)) &
                                (df_long['Date'] <= pd.Timestamp(end_date_dca))]


        if normalize:
            normalized_df_list = []
            for commodity in commodities:
                commodity_df = filtered_df_long[filtered_df_long['Commodity'] == commodity].copy()
                if not commodity_df.empty: # Check if dataframe is not empty
                    starting_price = commodity_df['Price'].iloc[0]
                    if starting_price != 0: # Avoid division by zero
                        commodity_df['Price'] = (commodity_df['Price'] / starting_price) * 100
                        normalized_df_list.append(commodity_df)
            if normalized_df_list: # Check if list is not empty
                filtered_df_long = pd.concat(normalized_df_list)
            else:
                 st.warning("Could not normalize prices (possibly zero starting price or no data).")


        fig = px.line(filtered_df_long, x='Date', y='Price', color='Commodity',
                    title=f'Price Evolution of {", ".join(commodities)}')
        st.plotly_chart(fig)
        st.write("Data Source: [https://fcainfoweb.nic.in/reports/report_menu_web.aspx](%s)" % url_dca)

    with col2:
        st.subheader("Week-on-Week Momentum (%)")
        # Ensure filtered_df_long is not empty before proceeding
        if not filtered_df_long.empty:
            latest_5_weeks_df = filtered_df_long.groupby('Commodity').apply(lambda x: x.tail(5)).reset_index(drop=True)
            pct_change_table_data = []
            for commodity in commodities:
                commodity_df = latest_5_weeks_df[latest_5_weeks_df['Commodity'] == commodity].copy()
                if len(commodity_df) > 1: # Need at least 2 data points for pct_change
                    commodity_df['Pct_Change'] = commodity_df['Price'].pct_change() * 100
                    pct_change_table_data.append(commodity_df.tail(4)) # Get last 4 changes

            if pct_change_table_data:
                pct_change_df = pd.concat(pct_change_table_data)[['Commodity', 'Date', 'Pct_Change']].dropna()
                if not pct_change_df.empty:
                    pct_change_df['Date'] = pd.to_datetime(pct_change_df['Date'])
                    pct_change_wide = pct_change_df.pivot(index='Commodity', columns='Date', values='Pct_Change')

                    # Calculate cumulative momentum
                    cumulative_momentum = []
                    for commodity_idx in pct_change_wide.index:
                        momentum_values = pct_change_wide.loc[commodity_idx].dropna().values # Drop NaNs before calculation
                        if len(momentum_values) > 0:
                            cum_momentum = np.prod([(1 + m/100) for m in momentum_values]) - 1
                            cumulative_momentum.append(cum_momentum * 100)
                        else:
                            cumulative_momentum.append(np.nan) # Append NaN if no valid momentum values

                    pct_change_wide = pct_change_wide.sort_index(axis=1, ascending=True)

                    num_weeks = len(pct_change_wide.columns)
                    column_names = ['Three weeks ago', 'Two weeks ago', 'Previous week', 'Latest week'][-num_weeks:] if num_weeks <= 4 else [f'Week {i+1}' for i in range(num_weeks)]
                    column_dates = pct_change_wide.columns.strftime('%d-%m-%y')

                    new_column_names = [f'{name}\n({date})' for name, date in zip(column_names, column_dates)]
                    pct_change_wide.columns = new_column_names

                    # Add cumulative momentum column
                    pct_change_wide[f'{min(num_weeks, 4)}-Week\nMomentum (%)'] = cumulative_momentum # Adjust label based on actual weeks

                    pct_change_wide = pct_change_wide.reset_index()
                    pct_change_wide = pct_change_wide.round(2)

                    st.markdown(pct_change_wide.to_html(escape=False, index=False), unsafe_allow_html=True) # Use index=False
                    st.write("Note: \n 1. Week-on-week momentum is calculated as per cent change in average weekly prices (excl. weekends).\n 2. As per latest data available on DoCA website.")
                else:
                    st.write("Not enough data points for momentum calculation in the selected period.")
            else:
                st.write("No percentage change data available for the selected period and commodities.")
        else:
             st.write("No data available for the selected period and commodities to calculate momentum.")


# --- Tab 2: Rainfall Deviation ---
with tab2:
    st.header("Rainfall Deviation")
    # Assuming get_rainfall_labels and fetch_rainfall_data are defined in utils.helper or globally
    try:
        rainfall_labels = get_rainfall_labels()
        col1, col2 = st.columns(2)
        with col1:
            url = "https://mausam.imd.gov.in/responsive/rainfallinformation_state.php"
            rainfall_type = st.selectbox(
                "Select Period",
                options=[
                    {'label': rainfall_labels[0], 'value': 'D'},
                    {'label': rainfall_labels[1], 'value': 'W'},
                    {'label': rainfall_labels[2], 'value': 'M'},
                    {'label': rainfall_labels[3], 'value': 'C'}
                ],
                format_func=lambda x: x['label'],
                index=2,
                key="rainfall_period"
            )['value']

            df_rain = fetch_rainfall_data(rainfall_type) # Assuming this function exists

            if df_rain is not None and india_geojson is not None:
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
                fig.update_geos(fitbounds="locations", visible=False, projection_scale=5.5, center={"lat": 20.5937, "lon": 78.9629})
                fig.update_layout(title_font=dict(size=20, family='Arial'), title_x=0, title_y=0.95, margin={"r":0,"t":0,"l":0,"b":0}, height=500, coloraxis_colorbar=dict(title="Deviation (%)", title_font=dict(size=14, family='Arial')))
                st.plotly_chart(fig)
            else:
                st.warning("Could not display rainfall map. Rainfall data or GeoJSON might be missing.")

        with col2:
            st.subheader("Major Crop Producers | Statewise")
            url_major = "https://upag.gov.in/"
            selected_commodity_rainfall = st.selectbox("Select a commodity:", ["None"] + list_of_crops, index=1, key="rainfall_commodity")

            if selected_commodity_rainfall != "None":
                try:
                    # Assuming fetch_major_producers is defined in utils.helper or globally
                    df_producers = fetch_major_producers(selected_commodity_rainfall)
                    if df_producers is not None:
                        st.dataframe(df_producers.round(1))
                    else:
                        st.warning("Production data for the latest year is not available.")
                except Exception as e:
                    st.warning("Production data for the latest year is not available.")
            st.write("Data Source: [https://upag.gov.in/](%s)" % url_major)

        if df_rain is not None:
            st.subheader("Rainfall Data")
            st.dataframe(df_rain.round(1))
            st.write("Data Source: [https://mausam.imd.gov.in/responsive/rainfallinformation_state.php](%s)" % url)
        else:
            st.warning("Rainfall data table cannot be displayed.")

    except NameError as e:
        st.error(f"Error in Rainfall Tab: A required function (like get_rainfall_labels, fetch_rainfall_data, fetch_major_producers) might be missing. Details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred in the Rainfall Tab: {e}")


# --- Tab 3: Food Production Trends ---
with tab3:
    st.header("Agriculture Crops Production")

    # Ensure agri_prod_long is loaded
    if agri_prod_long is not None:
        agri_prod_totals = agri_prod_long[agri_prod_long['Season'] == 'Total']
        agri_prod_seasonal = agri_prod_long[agri_prod_long['Season'] != 'Total']

        plot_col1_agri, plot_col2_agri = st.columns(2)

        with plot_col1_agri:
            sorted_crops_agri = sorted(agri_prod_seasonal['Crop'].unique())
            # Handle potential IndexError if 'Rice' is not in the list
            default_ind_agri = sorted_crops_agri.index('Rice') if 'Rice' in sorted_crops_agri else 0

            selected_crop_agri = st.selectbox('Select an agriculture crop', options=sorted_crops_agri, index=default_ind_agri, key="agri_crop_select")
            filtered_df_agri = agri_prod_seasonal[agri_prod_seasonal['Crop'] == selected_crop_agri]
            fig1_agri = px.bar(filtered_df_agri, x='Year', y='Value', color='Season',
                            title=f'Production Trend for {selected_crop_agri}', height=600,
                            labels = {'Year': 'Year', 'Value': "Production (in lakh tonnes)"})
            st.plotly_chart(fig1_agri, use_container_width=True)

        with plot_col2_agri:
            # Ensure 'Season' column exists and has unique values
            if 'Season' in agri_prod_long.columns:
                 season_options = sorted(agri_prod_long['Season'].unique())
                 selected_season_agri = st.selectbox(
                     'Select season for Y-o-Y analysis',
                     options=season_options,
                     index=0, # Default to the first season option
                     key="agri_season_select"
                 )

                 yoy_data_agri = agri_prod_long[
                     (agri_prod_long['Crop'] == selected_crop_agri) &
                     (agri_prod_long['Season'] == selected_season_agri)
                 ].sort_values('Year')

                 if not yoy_data_agri.empty:
                     latest_year_agri = yoy_data_agri['Year'].max()

                     # Validation for Total season
                     if selected_season_agri == 'Total':
                         crop_seasons_agri = agri_prod_seasonal[
                             agri_prod_seasonal['Crop'] == selected_crop_agri
                         ]['Season'].unique()

                         seasonal_data_check_agri = agri_prod_seasonal[
                             (agri_prod_seasonal['Crop'] == selected_crop_agri) &
                             (agri_prod_seasonal['Year'] == latest_year_agri)
                         ]

                         seasons_with_nan_agri = seasonal_data_check_agri[
                             seasonal_data_check_agri['Value'].isna()
                         ]['Season'].tolist()

                         if seasons_with_nan_agri:
                             st.warning(
                                 f"Data for 'Total' season in the latest year is incomplete. "
                                 f"Missing data for: {', '.join(seasons_with_nan_agri)}. "
                                 "Excluding latest year from Y-o-Y calculation."
                             )
                             yoy_data_agri = yoy_data_agri[yoy_data_agri['Year'] < latest_year_agri]

                     # Calculate YoY changes if data remains
                     if not yoy_data_agri.empty and len(yoy_data_agri) > 1:
                         yoy_data_agri['YoY_Change'] = yoy_data_agri['Value'].pct_change().mul(100).round(2)
                         plot_data_agri = yoy_data_agri.tail(10)

                         fig2_agri = px.bar(
                             plot_data_agri,
                             x='Year',
                             y='YoY_Change',
                             title=f'Y-o-Y change (%) in production of {selected_crop_agri} ({selected_season_agri})',
                             labels={'YoY_Change': 'Y-o-Y Change (%)'},
                             height=600
                         )
                         fig2_agri.update_traces(textposition='outside', text=plot_data_agri['YoY_Change'].round(1)) # Add text labels
                         st.plotly_chart(fig2_agri, use_container_width=True)
                     elif len(yoy_data_agri) <= 1:
                          st.info("Not enough data points (need more than 1 year) for YoY analysis.")
                     else: # yoy_data_agri became empty after validation
                         st.info("No complete data available for YoY analysis after validation.")
                 else:
                     st.warning("No data available for selected crop and season combination")
            else:
                 st.warning("Season information not found in agriculture production data.")


        url_agri = "https://agriwelfare.gov.in/en/AgricultureEstimates"
        st.write("Data Source: [https://agriwelfare.gov.in/en/AgricultureEstimates](%s)" % url_agri)

        # --- Horticulture Section ---
        st.header("Horticulture Crops Production")
        if horti_long is not None:
            # Assuming get_latest_and_change is defined in utils.helper or globally
            try:
                col1_horti, col2_horti, col3_horti = st.columns(3) # Renamed to avoid conflict
                # Note: Original code had 'Citrus Total', 'Total Fruits', 'Total Vegetables' which were excluded during loading.
                # Need to adjust or ensure these totals are present if metrics are desired.
                # Placeholder:
                fruits_latest, fruits_change = get_latest_and_change(horti_long, 'Total Fruits') # Example
                veg_latest, veg_change = get_latest_and_change(horti_long, 'Total Vegetables') # Example
                col1_horti.metric("Fruits Production", f"{fruits_latest/100000:.2f} lakh MT", f"{fruits_change:.2f}%") # Example
                col2_horti.metric("Vegetables Production", f"{veg_latest/10000:.2f} lakh MT", f"{veg_change:.2f}%") # Example
                st.info("Metrics for total Fruits/Vegetables require these categories in the loaded data.")

                plot_col1_horti, plot_col2_horti = st.columns(2) # Renamed

                with plot_col1_horti:
                    sorted_crops_horti = sorted(horti_long['Crops'].unique())
                    # Handle potential IndexError if 'Onion' is not in the list
                    default_index_horti = sorted_crops_horti.index('Onion') if 'Onion' in sorted_crops_horti else 0

                    selected_crop_horti = st.selectbox('Select a horticulture crop',
                                                    options=sorted_crops_horti,
                                                    index=default_index_horti,
                                                    key="horti_crop_select")
                    filtered_df_horti = horti_long[horti_long['Crops'] == selected_crop_horti].sort_values(by='Year')

                    fig1_horti = px.bar(filtered_df_horti, x='Year', y='Production_in_tonnes',
                                    title=f'Production Trend for {selected_crop_horti}',
                                    labels = {'Year': 'Year', 'Production_in_tonnes': "Production (in tonnes)"})

                    fig1_horti.add_scatter(x=filtered_df_horti['Year'], y=filtered_df_horti['Area'],
                                        mode='lines+markers', name='Area (RHS)', yaxis='y2')

                    fig1_horti.update_layout(
                        yaxis2=dict(title='Area (hectares)', overlaying='y', side='right', showgrid=False),
                        legend=dict(yanchor="top", y=1.09, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig1_horti, use_container_width=True)

                with plot_col2_horti:
                    if not filtered_df_horti.empty and len(filtered_df_horti) > 1:
                        filtered_df_horti['YoY_Change'] = filtered_df_horti['Production_in_tonnes'].pct_change() * 100
                        fig2_horti = px.bar(filtered_df_horti.tail(10), x='Year', y='YoY_Change',
                                        text=filtered_df_horti.tail(10)['YoY_Change'].round(1),
                                        title=f'Y-o-Y change (%) in production of {selected_crop_horti}',
                                        labels = {'Year': 'Year', 'YoY_Change': "Y-o-Y Change (%)"},
                                        color_discrete_sequence=['#1f77b4'])
                        fig2_horti.update_traces(textposition='outside')
                        st.plotly_chart(fig2_horti, use_container_width=True)
                    elif len(filtered_df_horti) <= 1:
                         st.info("Not enough data points (need more than 1 year) for YoY analysis.")
                    else:
                        st.warning("No data available for selected horticulture crop.")


                url_horti = "https://agriwelfare.gov.in/en/StatHortEst"
                st.write("Data Source: [https://agriwelfare.gov.in/en/StatHortEst](%s)" % url_horti)

            except NameError as e:
                 st.error(f"Error in Horticulture Section: A required function (like get_latest_and_change) might be missing. Details: {e}")
            except Exception as e:
                 st.error(f"An unexpected error occurred in the Horticulture Section: {e}")

        else:
            st.warning("Horticulture data not loaded. Skipping Horticulture section.")
    else:
        st.warning("Agriculture production data not loaded. Skipping Agriculture section.")


# --- Tab 5: Daily Mandi Arrivals ---
with tab4:
    st.header("Daily Mandi Arrivals Analysis")
    st.write("Source: https://agmarknet.gov.in/")

    # Load data
    mandi_data = load_daily_arrival_data()

    if mandi_data is not None:
        # Convert 'Date' column to datetime
        mandi_data['Date'] = pd.to_datetime(mandi_data['Date'], format='%d-%b-%Y', errors='coerce') # Coerce errors
        mandi_data = mandi_data.dropna(subset=['Date']) # Drop rows where date conversion failed

        # Commodity selection with capitalized names
        commodity_list_mandi = sorted(mandi_data['Commodity'].unique())
        # Find index for 'Onion' or default to 0
        default_commodity_index = commodity_list_mandi.index('ONION') if 'ONION' in commodity_list_mandi else 0

        selected_commodity_mandi = st.selectbox(
            "Select Commodity",
            commodity_list_mandi,
            index=default_commodity_index,
            format_func=lambda x: x.capitalize(), # Display names in capitalized format
            key="mandi_commodity"
        )

        # Get current date and default range
        max_mandi_date = mandi_data['Date'].max()
        current_date_mandi = max_mandi_date # Use max date from data as current
        current_year_mandi = current_date_mandi.year
        current_month_mandi = current_date_mandi.month

        start_of_current_month_mandi = current_date_mandi.replace(day=1)
        end_date_default_mandi = current_date_mandi

        # Date range selection for seasonality analysis
        col1_mandi, col2_mandi = st.columns(2)
        with col1_mandi:
            start_date_mandi = st.date_input("Start Date", start_of_current_month_mandi.date(), key="mandi_start_date", max_value=max_mandi_date.date())
        with col2_mandi:
            end_date_mandi = st.date_input("End Date", end_date_default_mandi.date(), key="mandi_end_date", max_value=max_mandi_date.date())

        # Filter data based on selections for seasonality analysis
        filtered_data_mandi = mandi_data[
            (mandi_data['Commodity'] == selected_commodity_mandi) &
            (mandi_data['Date'] >= pd.Timestamp(start_date_mandi)) &
            (mandi_data['Date'] <= pd.Timestamp(end_date_mandi))
        ]

        # Checkbox to toggle display of filtered data
        if st.checkbox("📄Show Selected Mandi Data", key="show_mandi_data"):
            st.subheader("Mandi Arrivals Data")
            st.write(mandi_data[(mandi_data['Commodity'] == selected_commodity_mandi)])

        # Create two columns for the charts
        col1_mandi_chart, col2_mandi_chart = st.columns(2)

        # Time Series Line Chart: January to December for the last 5 years
        with col1_mandi_chart:
            def get_monthly_sums_mandi(data, selected_commodity, current_year, current_month, end_date_actual):
                results = []
                # Go back 4 years from current year (total 5 years including current)
                for year in range(current_year - 4, current_year + 1):
                    for month in range(1, 13):
                        # Filter data for the specific year and month
                        monthly_data = data[
                            (data['Commodity'] == selected_commodity) &
                            (data['Date'].dt.year == year) &
                            (data['Date'].dt.month == month)
                        ]
                        # Ensure we don't include future data for the current year/month
                        if year == current_year and month == current_month:
                             monthly_data = monthly_data[monthly_data['Date'] <= end_date_actual]

                        total_value = monthly_data['Total Value'].sum() / 1000 # Convert to thousand tonnes
                        if total_value > 0: # Only include months with data
                            results.append({
                                'Year': year,
                                'Month': month,
                                'Total Value': total_value
                            })
                return pd.DataFrame(results)

            monthly_sums_mandi = get_monthly_sums_mandi(mandi_data, selected_commodity_mandi, current_year_mandi, current_month_mandi, max_mandi_date)

            if not monthly_sums_mandi.empty:
                fig_mandi_ts = px.line(
                    monthly_sums_mandi,
                    x='Month',
                    y='Total Value',
                    color='Year',
                    title=f'Monthly Arrival Trend for {selected_commodity_mandi.capitalize()}',
                    labels={'Month': 'Month', 'Total Value': 'Total Arrivals (thousand tonnes)'},
                    markers=True
                )
                fig_mandi_ts.update_xaxes(
                    tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
                st.plotly_chart(fig_mandi_ts, use_container_width=True)
            else:
                st.warning(f"No monthly arrival data found for {selected_commodity_mandi.capitalize()}.")


        # Seasonality Analysis: Bar chart for selected date range
        with col2_mandi_chart:
            def get_previous_years_data_mandi(data, start_date, end_date, selected_commodity, years_back):
                results = []
                start_date_ts = pd.Timestamp(start_date)
                end_date_ts = pd.Timestamp(end_date)
                for i in range(0, years_back + 1): # Include current year (i=0) up to years_back
                    prev_start = start_date_ts - pd.DateOffset(years=i)
                    prev_end = end_date_ts - pd.DateOffset(years=i)
                    # Ensure the comparison dates are valid within the data range
                    min_data_date = data['Date'].min()
                    if prev_start < min_data_date and prev_end < min_data_date:
                        continue # Skip if the entire period is before data starts

                    prev_data = data[
                        (data['Commodity'] == selected_commodity) &
                        (data['Date'] >= prev_start) &
                        (data['Date'] <= prev_end)
                    ].copy() # Use copy to avoid SettingWithCopyWarning
                    prev_data['Year'] = prev_start.year # Assign year based on the start of the period
                    results.append(prev_data)
                if not results:
                    return pd.DataFrame() # Return empty DataFrame if no data found
                return pd.concat(results)

            # Get data for the past 4 years + current year (total 5 years)
            past_years_data_mandi = get_previous_years_data_mandi(
                mandi_data,
                start_date_mandi,
                end_date_mandi,
                selected_commodity_mandi,
                years_back=4 # Look back 4 years
            )

            if not past_years_data_mandi.empty:
                seasonality_summary_mandi = past_years_data_mandi.groupby('Year')['Total Value'].sum().div(1000).reset_index()
                seasonality_summary_mandi = seasonality_summary_mandi.sort_values('Year')

                # Calculate YoY % change only if there are multiple years
                if len(seasonality_summary_mandi) > 1:
                    seasonality_summary_mandi['YoY Change (%)'] = seasonality_summary_mandi['Total Value'].pct_change() * 100
                else:
                    seasonality_summary_mandi['YoY Change (%)'] = np.nan # Assign NaN if only one year

                fig_mandi_bar = px.bar(
                    seasonality_summary_mandi,
                    x='Year',
                    y='Total Value',
                    title=f'Arrivals from {start_date_mandi} to {end_date_mandi} vs Previous Years',
                    color_discrete_sequence=['#9467bd']
                )

                # Add YoY % change as text labels (only if calculated)
                if 'YoY Change (%)' in seasonality_summary_mandi.columns and len(seasonality_summary_mandi) > 1:
                     fig_mandi_bar.add_trace(
                         go.Scatter(
                             x=seasonality_summary_mandi['Year'][1:],
                             y=seasonality_summary_mandi['Total Value'][1:],
                             mode='text',
                             text=[f"{val:.1f}%" if pd.notnull(val) else "" for val in seasonality_summary_mandi['YoY Change (%)'][1:]],
                             textposition="top center",
                             textfont=dict(color="black", size=12), # Changed color for visibility
                             showlegend=False
                         )
                     )


                fig_mandi_bar.update_xaxes(
                    tickvals=seasonality_summary_mandi['Year'],
                    ticktext=seasonality_summary_mandi['Year'].astype(str)
                )
                fig_mandi_bar.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Total Arrivals (thousand tonnes)",
                    showlegend=False,
                    template="plotly_white"
                )
                st.plotly_chart(fig_mandi_bar, use_container_width=True)

                # --- YoY Table for All Commodities ---
                commodity_yoy_list = []
                all_mandi_commodities = mandi_data['Commodity'].unique()

                with st.spinner("Calculating YoY changes for all commodities..."):
                    for commodity in all_mandi_commodities:
                        try:
                            past_data_comm = get_previous_years_data_mandi(
                                mandi_data, start_date_mandi, end_date_mandi, commodity, years_back=4
                            )
                            if not past_data_comm.empty:
                                summary_comm = past_data_comm.groupby('Year')['Total Value'].sum().div(1000).reset_index()
                                summary_comm = summary_comm.sort_values('Year')
                                if len(summary_comm) > 1:
                                    summary_comm['YoY Change (%)'] = summary_comm['Total Value'].pct_change() * 100
                                    latest_yoy = summary_comm.iloc[-1]['YoY Change (%)']
                                    commodity_yoy_list.append({
                                        'Commodity': commodity.capitalize(),
                                        'YoY Change (%)': round(latest_yoy, 1) if pd.notnull(latest_yoy) else None
                                    })
                                else:
                                     commodity_yoy_list.append({'Commodity': commodity.capitalize(), 'YoY Change (%)': None})
                            else:
                                commodity_yoy_list.append({'Commodity': commodity.capitalize(), 'YoY Change (%)': None})
                        except Exception as e:
                            commodity_yoy_list.append({'Commodity': commodity.capitalize(), 'YoY Change (%)': f"Error: {e}"}) # Log error

                if commodity_yoy_list:
                    yoy_df = pd.DataFrame(commodity_yoy_list)
                    yoy_df_sorted = yoy_df.sort_values(by='YoY Change (%)', ascending=False, na_position='last') # Handle NaNs
                    st.subheader(f"Y-o-Y % change in arrivals from {start_date_mandi} to {end_date_mandi} (Latest Year)")
                    st.dataframe(yoy_df_sorted.reset_index(drop=True))
                else:
                    st.info("Could not calculate YoY changes for all commodities.")

            else:
                st.warning(f"No arrival data found for {selected_commodity_mandi.capitalize()} in the selected or previous corresponding periods.")

    else:
        st.error("Mandi arrival data (commodity_data.csv) could not be loaded. Skipping this tab.")


# --- Tab: Global Inflation (NEW) ---
with tab_global_inflation:
    st.header("Monthly Headline Inflation Rates")
    st.subheader("CPI Y-o-Y (%)")
    # st.markdown("""
    #     This heatmap visualizes inflation rates over time for various countries.
    #     - **Colors:** Scaled within each country based on its *entire* historical data (Green=Low, Red=High).
    #     - **Labels:** Show the actual inflation value for the specific month.
    #     Use the date pickers below to select the desired time period for display.
    # """)

    # --- Load Global Inflation Data ---
    # Use the specific loading function defined earlier
    df_inflation = load_inflation_data(file_path="inflation_long.csv") # Ensure path is correct

    if df_inflation is not None:
        # --- Process Global Inflation Data ---
        df_pivot_inflation_full = pivot_inflation_data(df_inflation)
        df_normalized_inflation_full = normalize_inflation_full_history(df_pivot_inflation_full)

        if df_pivot_inflation_full is not None and df_normalized_inflation_full is not None:
            min_date_inflation = df_inflation['Date'].min().date()
            max_date_inflation = df_inflation['Date'].max().date()

            # Calculate default start date (15 months ago including the start month)
            default_start_date_inf = (pd.to_datetime(max_date_inflation) - pd.DateOffset(months=14)).date()
            # Ensure default start date is not before min_date
            default_start_date_inf = max(min_date_inflation, default_start_date_inf)

            # --- Date Selection UI ---
            col1_inf, col2_inf = st.columns(2)
            with col1_inf:
                selected_start_date_inf = st.date_input(
                    "Start date",
                    value=default_start_date_inf,
                    min_value=min_date_inflation,
                    max_value=max_date_inflation,
                    key="heatmap_start_date" # Unique key
                )
            with col2_inf:
                selected_end_date_inf = st.date_input(
                    "End date",
                    value=max_date_inflation,
                    min_value=min_date_inflation,
                    max_value=max_date_inflation,
                    key="heatmap_end_date" # Unique key
                )

            # --- Filtering and Plotting ---
            if selected_start_date_inf > selected_end_date_inf:
                st.error("Error: End date must fall after start date.")
            else:
                start_timestamp_inf = pd.to_datetime(selected_start_date_inf)
                end_timestamp_inf = pd.to_datetime(selected_end_date_inf)

                # Filter columns based on selected date range
                valid_cols_inf = (df_pivot_inflation_full.columns >= start_timestamp_inf) & (df_pivot_inflation_full.columns <= end_timestamp_inf)
                date_columns_to_display_inf = df_pivot_inflation_full.columns[valid_cols_inf]

                if date_columns_to_display_inf.empty:
                    st.warning(f"No global inflation data available for the selected period: {selected_start_date_inf.strftime('%Y-%m-%d')} to {selected_end_date_inf.strftime('%Y-%m-%d')}.")
                else:
                    # Select the subset period from BOTH normalized and original pivoted data
                    df_normalized_subset_inf = df_normalized_inflation_full[date_columns_to_display_inf]
                    df_pivot_subset_inf = df_pivot_inflation_full[date_columns_to_display_inf] # For annotations

                    # Handle potential all-NaN rows/columns introduced by subsetting/normalization
                    df_normalized_subset_inf = df_normalized_subset_inf.dropna(axis=0, how='all').dropna(axis=1, how='all')
                    if not df_normalized_subset_inf.empty:
                         df_pivot_subset_inf = df_pivot_subset_inf.reindex_like(df_normalized_subset_inf) # Align annotation data
                    else:
                         df_pivot_subset_inf = pd.DataFrame() # Assign empty if subset becomes empty

                    # Generate and display the plot
                    heatmap_fig = create_inflation_heatmap(
                        df_normalized_subset_inf,
                        df_pivot_subset_inf,
                        selected_start_date_inf.strftime('%Y-%m-%d'),
                        selected_end_date_inf.strftime('%Y-%m-%d')
                    )

                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig) # Use st.pyplot for matplotlib figures
        else:
            st.error("Failed to process global inflation data for heatmap.")
    else:
        # Error message already shown by load_inflation_data
        st.warning("Skipping Global Inflation tab content.")
    
    df_to_display = df_pivot_subset_inf.T.round(2)
    df_to_display.index = pd.to_datetime(df_to_display.index)
    df_to_display.index = df_to_display.index.strftime("%b-%Y")
    
    if st.checkbox("📊 Show Selected Data"):
        st.subheader("Global Headline Inflation Data")
        st.dataframe(df_to_display)
    
    st.markdown("Data Source: CEIC")

with tab_spf_plot:
    render_spf_expectations_tab(csv_path="latest_spf_data.csv") 

# #--- Footer (Optional) ---
# st.markdown("""
#     <p style="font-size: 14px;">
#         <strong>Designed and developed by</strong><br>
#         <strong><span style="color: coral;">Prices and Monetary Research Division, DEPR</span></strong>
#     </p>
# """, unsafe_allow_html=True)
