"""
Data analysis module for FI RV Strategy Backtester.
Contains functions for data fetching, processing, and statistical calculations.
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar


def fetch_treasury_data(tenor, min_date, max_date):
    """
    Fetch treasury yield data from FRED.
    
    Parameters:
    - tenor: str, treasury tenor ('2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y')
    - min_date: datetime, start date for data fetch
    - max_date: datetime, end date for data fetch
    
    Returns:
    - pd.DataFrame: DataFrame with observation_date and yield columns
    """
    # Map tenor to FRED series ID
    tenor_map = {
        '2Y': 'DGS2', '3Y': 'DGS3', '5Y': 'DGS5', '7Y': 'DGS7',
        '10Y': 'DGS10', '20Y': 'DGS20', '30Y': 'DGS30'
    }
    
    tenor_upper = tenor.upper()
    series_id = tenor_map[tenor_upper]
    
    # Fetch data from FRED
    dataframe = pdr.DataReader(series_id, 'fred', min_date, max_date)
    df_copy = pd.DataFrame({
        'observation_date': dataframe.index,
        series_id: dataframe[series_id].values
    })
    df_copy['observation_date'] = pd.to_datetime(df_copy['observation_date'])
    df_copy = df_copy.dropna(subset=[series_id]).reset_index(drop=True)
    
    return df_copy, series_id


def calculate_matrices(df_copy, df_filtered, yield_col, days_window):
    """
    Calculate change, Sharpe, and hit rate matrices.
    
    Parameters:
    - df_copy: pd.DataFrame, full yield data
    - df_filtered: pd.DataFrame, filtered data for specific dates
    - yield_col: str, column name for yield data
    - days_window: int, number of days before and after each reference date
    
    Returns:
    - tuple: (change_matrix, sharpe_matrix, hit_rate_matrix, trade_details, normalized_series)
    """
    n_dates = len(df_filtered)
    window_size = 2 * days_window + 1
    
    change_matrix = np.zeros((window_size, window_size))
    count_matrix = np.zeros((window_size, window_size))
    trade_details = {}
    normalized_series = {}
    
    # Calculate matrices
    for i in range(n_dates):
        current_date = df_filtered.loc[i, 'observation_date']
        current_idx = df_copy[df_copy['observation_date'] == current_date].index[0]
        
        # Normalized time series
        offsets, yields = [], []
        for offset in range(-days_window, days_window + 1):
            idx = current_idx + offset
            if 0 <= idx < len(df_copy):
                offsets.append(offset)
                yields.append(df_copy.loc[idx, yield_col])
        
        if len(yields) > 0:
            reference_yield = df_copy.loc[current_idx, yield_col]
            normalized_yields = [y - reference_yield for y in yields]
            normalized_series[current_date] = (offsets, normalized_yields)
        
        # Calculate changes
        for start_offset in range(-days_window, days_window + 1):
            start_idx = current_idx + start_offset
            if 0 <= start_idx < len(df_copy):
                start_yield = df_copy.loc[start_idx, yield_col]
                start_date_val = df_copy.loc[start_idx, 'observation_date']
                
                for end_offset in range(-days_window, days_window + 1):
                    end_idx = current_idx + end_offset
                    if 0 <= end_idx < len(df_copy):
                        end_yield = df_copy.loc[end_idx, yield_col]
                        end_date_val = df_copy.loc[end_idx, 'observation_date']
                        change = end_yield - start_yield
                        
                        row, col = start_offset + days_window, end_offset + days_window
                        
                        if row < col:
                            key = (row, col)
                            if key not in trade_details:
                                trade_details[key] = []
                            
                            trade_details[key].append({
                                'reference_date': current_date,
                                'start_date': start_date_val,
                                'start_yield': start_yield,
                                'end_date': end_date_val,
                                'end_yield': end_yield,
                                'change': change
                            })
                            
                            change_matrix[row, col] += change
                            count_matrix[row, col] += 1
    
    # Average changes
    change_matrix = np.divide(change_matrix, count_matrix, where=count_matrix > 0, 
                             out=np.full_like(change_matrix, np.nan))
    
    # Sharpe ratio
    sharpe_matrix = np.full((window_size, window_size), np.nan)
    for i in range(window_size):
        for j in range(window_size):
            if i < j:
                key = (i, j)
                if key in trade_details and len(trade_details[key]) > 0:
                    changes = [detail['change'] for detail in trade_details[key]]
                    mean_change = np.mean(changes)
                    std_change = np.std(changes, ddof=1) if len(changes) > 1 else 0
                    if std_change > 0:
                        sharpe_matrix[i, j] = mean_change / std_change
    
    # Hit rate
    hit_rate_matrix = np.full((window_size, window_size), np.nan)
    for i in range(window_size):
        for j in range(window_size):
            if i < j:
                key = (i, j)
                if key in trade_details and len(trade_details[key]) > 0:
                    changes = [detail['change'] for detail in trade_details[key]]
                    positive_count = sum(1 for change in changes if change > 0)
                    hit_rate_matrix[i, j] = positive_count / len(changes)
    
    # Mask lower triangle
    for i in range(window_size):
        for j in range(window_size):
            if i >= j:
                change_matrix[i, j] = np.nan
                sharpe_matrix[i, j] = np.nan
                hit_rate_matrix[i, j] = np.nan
    
    return change_matrix, sharpe_matrix, hit_rate_matrix, trade_details, normalized_series


def create_heatmap(matrix, axis_labels, title, colorbar_title, tenor_upper, n_dates, 
                   text_array, value_format, hovertemplate):
    """
    Create a heatmap figure.
    
    Parameters:
    - matrix: np.array, data matrix to plot
    - axis_labels: list, labels for x and y axes
    - title: str, plot title
    - colorbar_title: str, title for colorbar
    - tenor_upper: str, treasury tenor
    - n_dates: int, number of dates analyzed
    - text_array: np.array, text to display on heatmap
    - value_format: str, format string for values
    - hovertemplate: str, template for hover text
    
    Returns:
    - go.Figure: Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=axis_labels,
        y=axis_labels,
        colorscale='RdYlGn',
        colorbar=dict(title=colorbar_title),
        text=text_array,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},
        hovertemplate=hovertemplate
    ))
    fig.update_layout(
        title=f'{title}<br>{n_dates} specific dates',
        xaxis_title='End Days Offset',
        yaxis_title='Start Days Offset',
        yaxis=dict(autorange='reversed'),
        template='plotly_dark',
        height=700
    )
    return fig


def create_time_series_plot(normalized_series, tenor_upper, n_dates, days_window):
    """
    Create normalized time series plot.
    
    Parameters:
    - normalized_series: dict, mapping dates to (offsets, yields)
    - tenor_upper: str, treasury tenor
    - n_dates: int, number of dates
    - days_window: int, days window
    
    Returns:
    - go.Figure: Plotly figure object
    """
    fig = go.Figure()
    for current_date, (offsets, normalized_yields) in normalized_series.items():
        fig.add_trace(go.Scatter(
            x=offsets,
            y=[y * 100 for y in normalized_yields],
            mode='lines+markers',
            name=current_date.strftime('%Y-%m-%d'),
            hovertemplate='%{x} days<br>%{y:.1f} bps<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(
        title=f'Normalized UST {tenor_upper} Yield Time Series<br>{n_dates} specific dates',
        xaxis_title='Days Offset from Reference Date',
        yaxis_title='Yield Change (basis points)',
        template='plotly_dark',
        height=600,
        hovermode='closest'
    )
    return fig


def calculate_table_data(sharpe_matrix, hit_rate_matrix, trade_details, days_window):
    """
    Calculate statistics for tables.
    
    Parameters:
    - sharpe_matrix: np.array, Sharpe ratio matrix
    - hit_rate_matrix: np.array, hit rate matrix
    - trade_details: dict, detailed trade information
    - days_window: int, days window
    
    Returns:
    - dict: Dictionary with best_trade, worst_trade, and overall_stats
    """
    window_size = 2 * days_window + 1
    valid_sharpes = []
    
    for i in range(window_size):
        for j in range(window_size):
            if i < j and not np.isnan(sharpe_matrix[i, j]):
                key = (i, j)
                changes = [detail['change'] for detail in trade_details[key]]
                valid_sharpes.append({
                    'start_offset': i - days_window,
                    'end_offset': j - days_window,
                    'sharpe': sharpe_matrix[i, j],
                    'mean_change': np.mean(changes) * 100,
                    'hit_rate': hit_rate_matrix[i, j] * 100,
                    'num_trades': len(changes),
                    'trade_details': trade_details[key]
                })
    
    valid_sharpes.sort(key=lambda x: x['sharpe'], reverse=True)
    
    return {
        'best_trade': valid_sharpes[0] if valid_sharpes else None,
        'worst_trade': valid_sharpes[-1] if valid_sharpes else None,
        'overall_stats': valid_sharpes
    }


def generate_plots(days_window, tenor, specific_dates_str):
    """
    Generate all plots as Plotly figures.
    
    Parameters:
    - days_window: int, number of days before and after each reference date
    - tenor: str, treasury tenor
    - specific_dates_str: str, comma or newline separated dates
    
    Returns:
    - tuple: (fig1, fig2, fig3, fig4, table_data)
    """
    # Parse dates
    specific_dates = [date.strip() for date in specific_dates_str.replace('\n', ',').split(',') 
                     if date.strip()]
    
    tenor_upper = tenor.upper()
    
    # Calculate date range
    specific_dates_dt = pd.to_datetime(specific_dates)
    min_date = specific_dates_dt.min() - pd.Timedelta(days=days_window + 10)
    max_date = specific_dates_dt.max() + pd.Timedelta(days=days_window + 10)
    
    # Fetch data
    df_copy, yield_col = fetch_treasury_data(tenor, min_date, max_date)
    
    # Filter for specific dates
    df_filtered = df_copy[df_copy['observation_date'].isin(specific_dates_dt)].reset_index(drop=True)
    n_dates = len(df_filtered)
    window_size = 2 * days_window + 1
    
    # Calculate matrices
    change_matrix, sharpe_matrix, hit_rate_matrix, trade_details, normalized_series = \
        calculate_matrices(df_copy, df_filtered, yield_col, days_window)
    
    # Create axis labels
    axis_labels = [str(i) for i in range(-days_window, days_window + 1)]
    
    # Create text arrays with empty strings for NaN values
    text_change = np.where(np.isnan(change_matrix), '', 
                          np.round(change_matrix * 100, 1).astype(str))
    text_sharpe = np.where(np.isnan(sharpe_matrix), '', 
                          np.round(sharpe_matrix, 2).astype(str))
    text_hit_rate = np.where(np.isnan(hit_rate_matrix), '', 
                            np.round(hit_rate_matrix * 100, 0).astype(str))
    
    # 1. Yield Change Heatmap
    fig1 = create_heatmap(
        change_matrix * 100,
        axis_labels,
        f'UST {tenor_upper} Yield Changes Heatmap',
        'Yield Change (bps)',
        tenor_upper,
        n_dates,
        text_change,
        '.1f',
        'Start: %{y}<br>End: %{x}<br>Change: %{z:.1f} bps<extra></extra>'
    )
    
    # 2. Sharpe Ratio Heatmap
    fig2 = create_heatmap(
        sharpe_matrix,
        axis_labels,
        f'UST {tenor_upper} Sharpe Ratio Heatmap',
        'Sharpe Ratio',
        tenor_upper,
        n_dates,
        text_sharpe,
        '.2f',
        'Start: %{y}<br>End: %{x}<br>Sharpe: %{z:.2f}<extra></extra>'
    )
    
    # 3. Hit Rate Heatmap
    fig3 = go.Figure(data=go.Heatmap(
        z=hit_rate_matrix * 100,
        x=axis_labels,
        y=axis_labels,
        colorscale='RdYlGn',
        colorbar=dict(title='Hit Rate (%)'),
        zmin=0,
        zmax=100,
        text=text_hit_rate,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},
        hovertemplate='Start: %{y}<br>End: %{x}<br>Hit Rate: %{z:.0f}%<extra></extra>'
    ))
    fig3.update_layout(
        title=f'UST {tenor_upper} Hit Rate Heatmap<br>{n_dates} specific dates',
        xaxis_title='End Days Offset',
        yaxis_title='Start Days Offset',
        yaxis=dict(autorange='reversed'),
        template='plotly_dark',
        height=700
    )
    
    # 4. Normalized Time Series
    fig4 = create_time_series_plot(normalized_series, tenor_upper, n_dates, days_window)
    
    # Calculate statistics for tables
    table_data = calculate_table_data(sharpe_matrix, hit_rate_matrix, trade_details, days_window)
    
    return fig1, fig2, fig3, fig4, table_data


def generate_date_preset(preset):
    """
    Generate dates based on the selected preset for 2025.
    
    Parameters:
    - preset: str, preset name
    
    Returns:
    - str: newline-separated dates in YYYY-MM-DD format
    """
    if preset == 'custom':
        return '''2025-01-31
2025-02-28
2025-03-31
2025-04-30
2025-05-31
2025-06-30
2025-07-31
2025-08-31'''
    
    dates = []
    
    if preset == 'end_of_month':
        # Last day of each month
        for month in range(1, 13):
            last_day = calendar.monthrange(2025, month)[1]
            dates.append(f"2025-{month:02d}-{last_day:02d}")
    
    elif preset == 'beginning_of_month':
        # First day of each month
        for month in range(1, 13):
            dates.append(f"2025-{month:02d}-01")
    
    elif preset == 'middle_of_month':
        # 15th of each month
        for month in range(1, 13):
            dates.append(f"2025-{month:02d}-15")
    
    elif preset == 'every_friday':
        # Every Friday in 2025
        current = datetime(2025, 1, 1)
        while current.weekday() != 4:  # 4 = Friday
            current += timedelta(days=1)
        while current.year == 2025:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
    
    elif preset == 'every_monday':
        # Every Monday in 2025
        current = datetime(2025, 1, 1)
        while current.weekday() != 0:  # 0 = Monday
            current += timedelta(days=1)
        while current.year == 2025:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
    
    elif preset == 'first_friday':
        # First Friday of each month
        for month in range(1, 13):
            current = datetime(2025, month, 1)
            while current.weekday() != 4:
                current += timedelta(days=1)
            dates.append(current.strftime('%Y-%m-%d'))
    
    elif preset == 'last_friday':
        # Last Friday of each month
        for month in range(1, 13):
            last_day = calendar.monthrange(2025, month)[1]
            current = datetime(2025, month, last_day)
            while current.weekday() != 4:
                current -= timedelta(days=1)
            dates.append(current.strftime('%Y-%m-%d'))
    
    elif preset == 'biweekly_monday':
        # Every other Monday starting from first Monday
        current = datetime(2025, 1, 1)
        while current.weekday() != 0:
            current += timedelta(days=1)
        while current.year == 2025:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=14)
    
    elif preset == 'quarterly':
        # End of each quarter (Mar, Jun, Sep, Dec)
        dates = ['2025-03-31', '2025-06-30', '2025-09-30', '2025-12-31']
    
    return '\n'.join(dates)
