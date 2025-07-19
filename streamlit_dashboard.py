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
    horti_long = horti_long[~horti_long['Crops'].isin(["Fruits", 'Citrus', 'Vegetables','Spices'])]
    horti_long['Year'] = horti_long['Year'].apply(lambda x: x.strip()[:2] + x.strip()[-2:])
    horti_long.columns = ['Crops', 'Year', 'Area', 'Production_in_tonnes']
    
    return horti_long

# Load the GeoJSON file for Indian states
@st.cache_data
def load_geojson():
    with open('india_states.geojson', 'r') as f:
        return json.load(f)

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
    if not isinstance(date_obj, (dt.date, pd.Timestamp)):
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
        if isinstance(reference_date, dt.date) and not isinstance(reference_date, dt.datetime):
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
    st.markdown("Data pertains to forecasts released at the latest policy meeting, for the current quarter and 3 quarters ahead.")

    # --- Load Data ---
    df = load_spf_data(csv_path)

    if df is None:
        st.warning("Could not load SPF data. Please ensure 'latest_spf_data.csv' exists and is correctly formatted.")
        return # Stop execution for this tab if data loading fails

    # # --- Optional Raw Data Display ---
    # if st.checkbox("📊 Show loaded SPF data", key="show_spf_raw"):
    #     st.dataframe(df)

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
        
        # Display data table only for Quarterly
        if selected_type == 'Quarterly':
            if st.checkbox(f"📈 Show {selected_type} data table", key=f"show_spf_{selected_type.lower()}_table"):
                display_cols = ['indicator', 'period', 'formatted_period', 'value', 'last_update_time']
                display_cols = [col for col in display_cols if col in df_filtered.columns]  # Ensure columns exist
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

    # --- Optional Raw Data Display ---
    if st.checkbox("📊 Show loaded SPF data", key="show_spf_raw"):
        st.dataframe(df)
    

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

# --- Define Tabs ---
tab1, tab2, tab3, tab4, tab_global_inflation, tab_spf_plot = st.tabs([
    "DCA Retail Price Trends",
    "Rainfall Deviation",
    "Food Production Trends",
    "Daily Mandi Arrivals",
    "Global Inflation",
    "SPF Forecasts"
])

with tab1:
    st.header("DCA Retail Price Trends")
    url_dca = "https://fcainfoweb.nic.in/reports/report_menu_web.aspx"
    # st.write("Data Source: [https://fcainfoweb.nic.in/reports/report_menu_web.aspx](%s)" % url_dca)
    
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
        st.write("Data Source: [https://fcainfoweb.nic.in/reports/report_menu_web.aspx](%s)" % url_dca)

    
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

            st.write("Note: \n 1. Week-on-week momentum is calculated as per cent change in average weekly prices (excl. weekends).\n 2. As per latest data available on DoCA website.")

        else:
            st.write("No data available for the selected period and commodities.")
            

with tab2:
    st.header("Rainfall Deviation")
    col1, col2 = st.columns(2)
    rainfall_labels = get_rainfall_labels()
    with col1:
        url = "https://mausam.imd.gov.in/responsive/rainfallinformation_state.php"
        # st.write("Data Source: [https://mausam.imd.gov.in/responsive/rainfallinformation_state.php](%s)" % url)
        rainfall_type = st.selectbox(
            "Select Period",
            options=[
                {'label': rainfall_labels[0], 'value': 'D'},
                {'label': rainfall_labels[1], 'value': 'W'},
                {'label': rainfall_labels[2], 'value': 'M'},
                {'label': rainfall_labels[3], 'value': 'C'}
            ],
            format_func=lambda x: x['label'],
            index=2
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
        # st.write("Data Source: [https://mausam.imd.gov.in/responsive/rainfallinformation_state.php](%s)" % url)

    with col2:
        st.subheader("Major Crop Producers | Statewise")
        url_major = "https://upag.gov.in/"
        # st.write("Data Source: [https://upag.gov.in/](%s)" % url_major)
        
        # Add None as the first option
        selected_commodity = st.selectbox("Select a commodity:", ["None"] + list_of_crops, index=1)
        
        # Only fetch and display data if a commodity is selected
        if selected_commodity != "None":
            try:
                df = fetch_major_producers(selected_commodity)
                if df is not None:
                    st.dataframe(df.round(1))
                else:
                    st.warning("Production data for the latest year is not available.")
            except Exception as e:
                # Suppress the actual error and only show the user-friendly warning
                st.warning("Production data for the latest year is not available.")
                # If you want to log the error for debugging but not display it
                # You could use logging here or print to console
                # print(f"Debug - Error: {str(e)}")
        st.write("Data Source: [https://upag.gov.in/](%s)" % url_major)

        
    st.subheader("Rainfall Data")
    st.dataframe(df_rain.round(1))
    st.write("Data Source: [https://mausam.imd.gov.in/responsive/rainfallinformation_state.php](%s)" % url)

with tab3:
    st.header("Agriculture Crops Production")

    # Original data splitting remains unchanged
    agri_prod_totals = agri_prod_long[agri_prod_long['Season'] == 'Total']
    agri_prod_seasonal = agri_prod_long[agri_prod_long['Season'] != 'Total']

    plot_col1, plot_col2 = st.columns(2)

    # First chart (unchanged)
    with plot_col1:
        sorted_crops_agri = sorted(agri_prod_seasonal['Crop'].unique())
        default_ind_agri = sorted_crops_agri.index('Rice')

        selected_crop = st.selectbox('Select a crop', options=sorted_crops_agri, index=default_ind_agri)
        filtered_df = agri_prod_seasonal[agri_prod_seasonal['Crop'] == selected_crop]
        fig1 = px.bar(filtered_df, x='Year', y='Value', color='Season',
                        title=f'Production Trend for {selected_crop}', height=600,
                        labels = {'Year': 'Year', 'Value': "Production (in lakh tonnes)"})
        st.plotly_chart(fig1, use_container_width=True)

    # Second chart with season validation
    with plot_col2:
        selected_season = st.selectbox(
            'Select season for Y-o-Y analysis',
            options=sorted(agri_prod_long['Season'].unique()),
            index=0
        )

        # Filter data for selected crop and season
        yoy_data = agri_prod_long[
            (agri_prod_long['Crop'] == selected_crop) &
            (agri_prod_long['Season'] == selected_season)
        ].sort_values('Year')

        if not yoy_data.empty:
            latest_year = yoy_data['Year'].max()
            
            # Validation for Total season
            if selected_season == 'Total':
                # Get all seasons except Total for the crop
                crop_seasons = agri_prod_seasonal[
                    agri_prod_seasonal['Crop'] == selected_crop
                ]['Season'].unique()
                
                # Check if latest year has NaNs for any season
                seasonal_data_check = agri_prod_seasonal[
                    (agri_prod_seasonal['Crop'] == selected_crop) &
                    (agri_prod_seasonal['Year'] == latest_year)
                ]
                
                # Identify seasons with NaN values
                seasons_with_nan = seasonal_data_check[
                    seasonal_data_check['Value'].isna()
                ]['Season'].tolist()

                if seasons_with_nan:
                    st.warning(
                        f"Data for 'Total' season in the latest year is incomplete. " 
                        f"Missing data for: {', '.join(seasons_with_nan)}. "
                        "Excluding latest year from Y-o-Y calculation."
                    )
                    yoy_data = yoy_data[yoy_data['Year'] < latest_year]

            # Calculate YoY changes if data remains
            if not yoy_data.empty:
                yoy_data['YoY_Change'] = yoy_data['Value'].pct_change().mul(100).round(2)
                plot_data = yoy_data.tail(10)

                fig2 = px.bar(
                    plot_data, 
                    x='Year', 
                    y='YoY_Change',
                    text='YoY_Change',
                    title=f'Y-o-Y change (%) in production of {selected_crop} ({selected_season})',
                    labels={'YoY_Change': 'Y-o-Y Change (%)'},
                    height=600
                )

                # Customize data labels
                fig2.update_traces(textposition='outside')

                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No complete data available for YoY analysis")
        else:
            st.warning("No data available for selected crop and season combination")

    url_agri = "https://agriwelfare.gov.in/en/AgricultureEstimates"
    st.write("Data Source: [https://agriwelfare.gov.in/en/AgricultureEstimates](%s)" % url_agri)


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

    st.header("Horticulture Crops Production")
    
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

        fig1 = px.bar(filtered_df, x='Year', y='Production_in_tonnes', 
                    title=f'Production Trend for {selected_crop}',
                    labels = {'Year': 'Year', 'Production_in_tonnes': "Production (in tonnes)"})

        fig1.add_scatter(x=filtered_df['Year'], y=filtered_df['Area'], 
                        mode='lines+markers', name='Area (RHS)', yaxis='y2')

        fig1.update_layout(
            yaxis2=dict(title='Area (hectares)', overlaying='y', side='right', showgrid=False),
            legend=dict(
                yanchor="top",
                y=1.09,
                xanchor="right",
                x=0.99
            )
        )

        st.plotly_chart(fig1, use_container_width=True)

    with plot_col2:
        filtered_df['YoY_Change'] = filtered_df['Production_in_tonnes'].pct_change() * 100
        fig2 = px.bar(filtered_df.tail(10), x='Year', y='YoY_Change', 
                    text=filtered_df.tail(10)['YoY_Change'].round(1),
                    title=f'Y-o-Y change (%) in production of {selected_crop}',
                    labels = {'Year': 'Year', 'YoY_Change': "Y-o-Y Change (%)"},
                    color_discrete_sequence=['#1f77b4'])
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    url_horti = "https://agriwelfare.gov.in/en/StatHortEst"
    st.write("Data Source: [https://agriwelfare.gov.in/en/StatHortEst](%s)" % url_horti)

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

from plotly import graph_objs as go

with tab4:
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
    # st.write("Source: https://agmarknet.gov.in/")

    # # Filters section
    # st.header("Filters")

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

    # Create two columns for the charts
    col1, col2 = st.columns(2)

    # Time Series Line Chart: January to December for the last 3 years
    with col1:
        #st.subheader(f"Monthly Arrival Trend for {selected_commodity.capitalize()}")

        # Function to get monthly sums for January to December for the last 3 years
        def get_monthly_sums(data, selected_commodity, current_year, current_month):#, current_day):
            results = []
            for year in range(current_year - 4, current_year + 1):  # Last 3 years
                for month in range(1, 13):  # January to December
                    if year == current_year and month == current_month:
                        # For the current month, include data only up to the current day
                        monthly_data = data[
                            (data['Commodity'] == selected_commodity) & 
                            (data['Date'].dt.year == year) & 
                            (data['Date'].dt.month == month) & 
                            (data['Date'].dt.day <= end_date_default.day)
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
        monthly_sums = get_monthly_sums(data, selected_commodity, current_year, current_month)#, current_day)

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
        # Function to get the same date range for previous years
        def get_previous_years_data(data, start_date, end_date, selected_commodity, years_back):
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

        # Get data for the past 3 years + current year (total 5 years)
        past_years_data = get_previous_years_data(
            data, 
            pd.to_datetime(start_date), 
            pd.to_datetime(end_date), 
            selected_commodity, 
            years_back=3
        )

        # Sum arrivals for each year
        seasonality_summary = past_years_data.groupby('Year')['Total Value'].sum().div(1000).reset_index()

        # Sort years to ensure chronological order
        seasonality_summary = seasonality_summary.sort_values('Year')

        # Calculate YoY % change
        seasonality_summary['YoY Change (%)'] = seasonality_summary['Total Value'].pct_change() * 100
        print(seasonality_summary)

        # Bar chart for arrivals
        fig3 = px.bar(
            seasonality_summary, 
            x='Year', 
            y='Total Value', 
            title=f'Cumulative arrivals from {start_date} to {end_date} vis-a-vis previous years',
            color_discrete_sequence=['#9467bd']
        )

        # Add YoY % change as text labels on top of bars (from second year onwards)
        fig3.add_trace(
            go.Scatter(
                x=seasonality_summary['Year'][1:], 
                y=seasonality_summary['Total Value'][1:],  # Slightly above bar height
                mode='text',
                text=[
                    f"{val:.1f}%" if pd.notnull(val) else "" 
                    for val in seasonality_summary['YoY Change (%)'][1:]
                ],
                textposition="top center",
                textfont=dict(color="white", size=12),
                showlegend=False
            )
        )

        # Clean up x-axis
        fig3.update_xaxes(
            tickvals=seasonality_summary['Year'],
            ticktext=seasonality_summary['Year'].astype(str)
        )

        # Layout settings
        fig3.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Arrivals (thousand tonnes)",
            showlegend=False,
            template="plotly_white"
        )

        st.plotly_chart(fig3, use_container_width=True)
    
        # Create empty list to store yoy change values
    commodity_yoy_list = []

    # Loop through each commodity to calculate YoY %
    for commodity in commodity_list:
        try:
            past_years_data = get_previous_years_data(
                data, pd.to_datetime(start_date), pd.to_datetime(end_date), commodity, years_back=3
            )
            seasonality_summary = past_years_data.groupby('Year')['Total Value'].sum().div(1000).reset_index()
            seasonality_summary = seasonality_summary.sort_values('Year')
            seasonality_summary['YoY Change (%)'] = seasonality_summary['Total Value'].pct_change() * 100

            latest_yoy = seasonality_summary.iloc[-1]['YoY Change (%)']  # Latest year YoY
            commodity_yoy_list.append({
                'Commodity': commodity.capitalize(),
                'YoY Change (%)': round(latest_yoy, 1) if pd.notnull(latest_yoy) else None
            })
        except Exception as e:
            # Handle edge cases (e.g., not enough data for the commodity)
            commodity_yoy_list.append({
                'Commodity': commodity.capitalize(),
                'YoY Change (%)': None
            })

    # Checkbox to toggle display of filtered data
    show_data = st.checkbox("📊 Show time series data")

    # Display filtered data if checkbox is checked
    if show_data:
        st.subheader("Filtered Data")

        # display_data = filtered_data.copy()
        # display_data['Date'] = pd.to_datetime(display_data['Date']).dt.strftime('%d-%m-%Y')
        # st.write(display_data) 
        st.write(data[(data['Commodity'] == selected_commodity)])

    # Convert to dataframe
    yoy_df = pd.DataFrame(commodity_yoy_list)

    # Optional: sort by YoY% descending
    yoy_df_sorted = yoy_df.sort_values(by='YoY Change (%)', ascending=False)

    # Display table
    st.subheader(f"Y-o-Y % change in arrivals from {start_date} to {end_date} (Latest Year)")
    st.dataframe(yoy_df_sorted.reset_index(drop=True))

    st.write("Source: https://agmarknet.gov.in/")

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
    
    if st.checkbox("📊 Show selected data"):
        st.subheader("Global Headline Inflation Data")
        st.dataframe(df_to_display)
    
    st.markdown("Data Source: CEIC")

with tab_spf_plot:
    render_spf_expectations_tab(csv_path="latest_spf_data.csv") 

    st.markdown("Data Source: CEIC")

# st.markdown("""
#     <p style="font-size: 14px;">
#         <strong>Designed and developed by</strong><br>
#         <strong><span style="color: coral;">Prices and Monetary Research Division, DEPR</span></strong>
#     </p>
# """, unsafe_allow_html=True)
