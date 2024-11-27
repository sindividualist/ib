import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from fredapi import Fred
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import os

fred_api_key = os.getenv("FRED_API_KEY", "default_key")

# Ensure Streamlit runs in headless mode
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Streamlit Page Configuration
st.set_page_config(page_title="Indykator Bitkojniorza", layout="wide")

# Centered Title
st.markdown(
    """
    <style>
    .center-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 40px;
    }
    body {
        background: white !important;
        color: black;
    }
    </style>
    <div class="center-title">Indykator Bitkojniorza</div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Settings
st.sidebar.header("Ustawienia")
api_key = st.sidebar.text_input("Podaj klucz API (FRED)", placeholder="Wpisz klucz API...")
st.sidebar.markdown(
    """
    <p style="font-size: 12px;">
    <a href="https://fred.stlouisfed.org/docs/api/api_key.html" target="_blank">Przeczytaj o kluczu FRED API Tutaj</a>
    </p>
    """,
    unsafe_allow_html=True,
)

# Define slider date ranges
min_date = datetime(2015, 1, 1)
max_date = datetime(2045, 1, 1)

# Start Date Slider
start_date = st.sidebar.slider(
    "Data początkowa",
    min_value=min_date,
    max_value=max_date - pd.Timedelta(days=1),  # Ensure end date is after start date
    value=datetime(2015, 1, 1),
    format="YYYY-MM-DD"
)

# End Date Slider
end_date = st.sidebar.slider(
    "Data końcowa",
    min_value=min_date + pd.Timedelta(days=1),  # Ensure start date is before end date
    max_value=max_date,
    value=datetime(2026, 1, 1),
    format="YYYY-MM-DD"
)

# Presets
st.sidebar.subheader("Szybkie ustawienia")
if st.sidebar.button("2015-01-01 to 2026-01-01"):
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 1, 1)

if st.sidebar.button(f"{datetime.today().year - 1}-01-01 to {datetime.today().year + 1}-01-01"):
    start_date = datetime(datetime.today().year - 1, 1, 1)
    end_date = datetime(datetime.today().year + 1, 1, 1)

# Define dynamic presets
today = datetime.today()
preset_ranges = {
    "Ostatnie 5 lat": (today - timedelta(days=5 * 365), today + timedelta(days=365)),
    "Ostatni 1 rok": (today - timedelta(days=365), today + timedelta(days=365)),
    "Ostatni miesiąc": (today - timedelta(days=30), today + timedelta(days=30)),
    "Ostatni tydzień": (today - timedelta(days=7), today + timedelta(days=7)),
}

# Buttons for dynamic presets
if st.sidebar.button("Ostatnie 5 lat"):
    start_date, end_date = preset_ranges["Ostatnie 5 lat"]

if st.sidebar.button("Ostatni 1 rok"):
    start_date, end_date = preset_ranges["Ostatni 1 rok"]

if st.sidebar.button("Ostatni miesiąc"):
    start_date, end_date = preset_ranges["Ostatni miesiąc"]

if st.sidebar.button("Ostatni tydzień"):
    start_date, end_date = preset_ranges["Ostatni tydzień"]

# Validate Start and End Dates
if start_date >= end_date:
    st.sidebar.error("Data początkowa musi być wcześniejsza niż data końcowa.")

# Constants
halving_dates = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-10', 
                                '2028-03-06', '2032-01-29', '2035-12-23', '2039-11-16', 
                                '2043-10-11'])
x_min = start_date
x_max = end_date

# Fetch Bitcoin price data
@st.cache_data
def fetch_data():
    return yf.download('BTC-USD', start='2015-01-01', end=datetime.today().strftime('%Y-%m-%d'), interval='1d')

btc_data = fetch_data()
btc_data['Date'] = btc_data.index
btc_data.reset_index(drop=True, inplace=True)

# MVRV Z-Score calculation
circulating_supply = 19000000  # Adjust as needed
btc_data['Market Value'] = btc_data['Close'] * circulating_supply
btc_data['Realized Value'] = btc_data['Close'].rolling(window=200).mean() * circulating_supply
btc_data['MVRV Z-Score'] = (btc_data['Market Value'] - btc_data['Realized Value']) / btc_data['Realized Value'].std()

# Calculate daily returns
btc_data['Daily Return'] = btc_data['Close'].pct_change()

# 1. Volatility: Standard deviation of daily returns (using a rolling 30-day window as a proxy for fear)
btc_data['Volatility'] = btc_data['Daily Return'].rolling(window=30).std()

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

btc_data['RSI'] = calculate_rsi(btc_data)

# Calculate Indicators
btc_data['Meyer Multiple'] = btc_data['Close'] / btc_data['Close'].rolling(window=200).mean()
btc_data['MA111'] = btc_data['Close'].rolling(window=111).mean()
btc_data['MA350'] = btc_data['Close'].rolling(window=350).mean() * 2
btc_data['Momentum'] = btc_data['Close'] / btc_data['Close'].rolling(window=200).mean()
btc_data['Volatility'] = btc_data['Close'].pct_change().rolling(window=30).std()
btc_data['Fear_Greed_Index'] = 100 * (
    (btc_data['Momentum'] - 1) / (3 - 1) +
    (1 - btc_data['Volatility']) +
    (btc_data['RSI'] - 30) / (70 - 30)
) / 3
btc_data['Fear_Greed_Index'] = btc_data['Fear_Greed_Index'].clip(0, 100)

# Resample data to monthly frequency
btc_monthly = btc_data.resample('ME', on='Date').last()
btc_monthly['RSI'] = calculate_rsi(btc_monthly)
btc_monthly.reset_index(inplace=True)

# Generate halving months (16 to 18 months after each halving)
halving_months = []
for halving in halving_dates:
    for i in range(16, 19):
        halving_months.append(halving + pd.DateOffset(months=i))

# Calculate months until potential top
next_halving = halving_dates[-6]
months_until_top = (next_halving + pd.DateOffset(months=18) - pd.Timestamp(datetime.today())).days // 30

# Formatter for price
def format_price(price):
    if price >= 1_000_000:
        return f"{price / 1_000_000:.0f}M"
    elif price >= 1_000:
        return f"{price / 1_000:.0f}k"
    else:
        return f"{price:.0f}"

# Helper function to add halving event lines
def add_halving_lines(ax):
    for halving in halving_dates:
        ax.axvline(halving, color='red', linestyle=':', alpha=0.5, label='Halving' if halving == halving_dates[0] else "")

# Generate date marks for months and years
month_marks = pd.date_range(start=x_min, end=x_max, freq='MS')  # Monthly start
year_marks = pd.date_range(start=x_min, end=x_max, freq='YS')   # Yearly start

# Function to add month and year markers to a plot
def add_time_markers(ax):
    # Add monthly markers
    #for month in month_marks:
    #    ax.axvline(month, color='grey', linestyle=':', alpha=0.5)
    # Add yearly markers
    for year in year_marks:
        ax.axvline(year, color='black', linestyle='--', alpha=0.1)

# Mark the last data day on all charts
def mark_last_data_day(ax, last_date):
    ax.axvline(last_date, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Ostatni dzień danych')

last_date = btc_data['Date'].iloc[-1]

# Bitcoin Price Chart
#st.subheader("Bitcoin Price and Indicators")
#fig, ax = plt.subplots(figsize=(14, 4))
fig, axs = plt.subplots(6, 1, figsize=(28, 36), sharex=True)  # Update to 6 rows for new chart
ax=axs[0]
ax.plot(btc_data['Date'], btc_data['Close'], label='Cena USD', color='black')
ax.set_yscale('log')
ax.set_ylabel("Cena USD")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_price(x)))
ax.plot(btc_data['Date'], btc_data['MA111'], label='MA111', color='blue')
ax.plot(btc_data['Date'], btc_data['MA350'], label='MA350 (x2)', color='orange')
ax.set_xlim(x_min, x_max)
add_halving_lines(ax)
add_time_markers(ax)
mark_last_data_day(ax, last_date) 
ax.legend()

# Monthly RSI Chart
#st.subheader("Monthly RSI Chart")
ax1=axs[1]
ax1.axhspan(0, 30, color='green', alpha=0.2, label='Wykupiony (<30)')
ax1.axhspan(30, 70, color='gray', alpha=0.1, label='Neutralny (30-70)')
ax1.axhspan(70, 100, color='red', alpha=0.2, label='Wyprzedany (>70)')
ax1.plot(btc_monthly['Date'], btc_monthly['RSI'], label='Miesięczny RSI', color='black', linewidth=1.0)
ax1.set_ylabel("Miesnięczny RSI")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(0, 100)
add_halving_lines(ax1)
add_time_markers(ax1)
mark_last_data_day(ax1, last_date) 
ax1.legend()

# Fear and Greed Index Chart
#st.subheader("Fear and Greed Index")
ax2=axs[2]
ax2.axhspan(80, 100, color='red', alpha=0.2, label='Chciwość')
ax2.axhspan(0, 20, color='green', alpha=0.2, label='Strach')
ax2.plot(btc_data['Date'], btc_data['Fear_Greed_Index'], label='Indeks Chciwości i Strachu', color='black', linewidth=1.0)
ax2.set_ylabel("Chciwość i Strach")
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(0, 100)
add_halving_lines(ax2)
add_time_markers(ax2)
mark_last_data_day(ax2, last_date) 
ax2.legend()

# MM
#st.subheader("MM")
ax3=axs[3]
ax3.axhspan(2.4, 4, color='red', alpha=0.2, label='Przewartościowanie')
ax3.axhspan(0, 1, color='green', alpha=0.2, label='Niedowartościowanie')
ax3.plot(btc_data['Date'], btc_data['Meyer Multiple'], label='Meyer Multiple', color='black', linewidth=1.0)
ax3.set_ylabel("Meyer Multiple")
ax3.set_xlim(x_min, x_max)
ax3.set_ylim(0, 4)
add_halving_lines(ax3)
add_time_markers(ax3)
mark_last_data_day(ax3, last_date) 
ax3.legend()


# MVRV
ax4=axs[4]
ax4.plot(btc_data['Date'], btc_data['MVRV Z-Score'], color='black', linewidth=1.0)
ax4.set_ylim(-2, 3)
ax4.set_xlim(x_min, x_max)
ax4.set_ylabel('MVRV Z-Score')
ax4.axhspan(2, 4, color='red', alpha=0.2, label='Przewartościowanie')
ax4.axhspan(0, -2, color='green', alpha=0.2, label='Niedowartościowanie')
add_halving_lines(ax4)
add_time_markers(ax4)
mark_last_data_day(ax4, last_date) 
ax4.legend()

# M2 Money Supply and Fed Rate

if not api_key:
    api_key='b32272dc5d5c554832920725f78318c1'
    fred = Fred(api_key=api_key)
    m2_data = fred.get_series('M2SL', start='2015-01-01')
    fed_rate_data = fred.get_series('FEDFUNDS', start='2015-01-01')
    m2_data = m2_data.reindex(btc_data['Date'], method='ffill')
    fed_rate_data = fed_rate_data.reindex(btc_data['Date'], method='ffill')

    #st.subheader("M2 Money Supply and Fed Interest Rates")
    ax5=axs[5]

    # Plot M2 Money Supply
    ax5.plot(btc_data['Date'], m2_data, label='Podaż pieniądza M2', color='blue')
    ax5.set_yscale('log')
    ax5.set_ylabel("Podaż M2")
    ax5.set_xlim(x_min, x_max)
    ax5.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Plot Fed Rate
    ax5b = ax5.twinx()
    ax5b.plot(btc_data['Date'], fed_rate_data, label='Stopy Procentowe FED', color='black', linestyle='-')
    #ax4b.set_ylabel("Fed Rates %")

    # Hide y-axis values for Fed Rate
    ax5b.tick_params(axis='y', which='both', right=False, labelright=True)
    
    # Add halving lines
    add_halving_lines(ax5)
    add_time_markers(ax5)
    mark_last_data_day(ax5, last_date) 
    # Legends
    ax5.legend(loc='upper left')
    ax5b.legend(loc='upper right')

if api_key:
    fred = Fred(api_key=api_key)
    m2_data = fred.get_series('M2SL', start='2015-01-01')
    fed_rate_data = fred.get_series('FEDFUNDS', start='2015-01-01')
    m2_data = m2_data.reindex(btc_data['Date'], method='ffill')
    fed_rate_data = fed_rate_data.reindex(btc_data['Date'], method='ffill')

    #st.subheader("M2 Money Supply and Fed Interest Rates")
    ax5=axs[5]

    # Plot M2 Money Supply
    ax5.plot(btc_data['Date'], m2_data, label='Podaż pieniądza M2', color='blue')
    ax5.set_yscale('log')
    ax5.set_ylabel("Podaż M2")
    ax5.set_xlim(x_min, x_max)
    ax5.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Plot Fed Rate
    ax5b = ax5.twinx()
    ax5b.plot(btc_data['Date'], fed_rate_data, label='Stopy Procentowe FED', color='black', linestyle='-')
    #ax4b.set_ylabel("Fed Rates %")

    # Hide y-axis values for Fed Rate
    ax5b.tick_params(axis='y', which='both', right=False, labelright=True)
    
    # Add halving lines
    add_halving_lines(ax5)
    add_time_markers(ax5)
    mark_last_data_day(ax5, last_date) 
    # Legends
    #ax5.legend(loc='upper left')
    #ax5b.legend(loc='upper right')

# Highlight yellow and orange zones based on conditions
for i in range(len(btc_data) - 1):
    if (btc_data['RSI'].iloc[i] > 70 and 
        btc_data['Meyer Multiple'].iloc[i] > 1 and 
        btc_data['MVRV Z-Score'].iloc[i] >= 1):
        ax.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='yellow', alpha=0.3)
        ax1.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='yellow', alpha=0.3)
        ax2.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='yellow', alpha=0.3)
        ax3.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='yellow', alpha=0.3)
        ax4.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='yellow', alpha=0.3)
        ax5.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='yellow', alpha=0.3)
        
    if (btc_data['RSI'].iloc[i] >= 70 and 
        btc_data['Meyer Multiple'].iloc[i] > 2 and 
        btc_data['MVRV Z-Score'].iloc[i] >= 2):
        ax.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='red', alpha=0.3)
        ax1.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='red', alpha=0.3)
        ax2.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='red', alpha=0.3)
        ax3.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='red', alpha=0.3)
        ax4.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='red', alpha=0.3)
        ax5.axvspan(btc_data['Date'].iloc[i], btc_data['Date'].iloc[i + 1], color='red', alpha=0.3)

# Add text for months until potential top etc
ax.text(0.01, 0.95, f'Ostatni dzień pobranych danych: {datetime.today().strftime("%Y-%m-%d")}',
         transform=ax.transAxes, fontsize=12, color='red', ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

# Calculate days until the potential top (approx. 18 months after the next halving)
days_until_top = (next_halving + pd.DateOffset(months=18) - pd.Timestamp(datetime.today())).days

# Update the annotation text with the days until top and days until May 1st, 2025
ax.text(0.01, 0.80, f'Dni do potencjalnego szczytu: {days_until_top}',
         transform=ax.transAxes, fontsize=12, color='red', ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

# Highlight halving months on all charts
for ax in axs:
    for month in halving_months:
        ax.axvspan(month - pd.DateOffset(months=1), month + pd.DateOffset(months=1),
                   color='green', alpha=0.1)

st.pyplot(fig)
