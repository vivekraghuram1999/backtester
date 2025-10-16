from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import plotly.graph_objects as go

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("FI RV Strategy Backtester", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Parameters", className="card-title"),
                    
                    html.Label("Days Window:"),
                    dcc.Input(id='days-window', type='number', value=7, min=1, max=30, className="form-control mb-3"),
                    
                    html.Label("Treasury Tenor:"),
                    dcc.Dropdown(
                        id='tenor',
                        options=[
                            {'label': '2 Year', 'value': '2Y'},
                            {'label': '3 Year', 'value': '3Y'},
                            {'label': '5 Year', 'value': '5Y'},
                            {'label': '7 Year', 'value': '7Y'},
                            {'label': '10 Year', 'value': '10Y'},
                            {'label': '20 Year', 'value': '20Y'},
                            {'label': '30 Year', 'value': '30Y'}
                        ],
                        value='2Y',
                        className="mb-3",
                        style={'color': 'black'}
                    ),
                    
                    html.Label("Date Preset:"),
                    dcc.Dropdown(
                        id='date-preset',
                        options=[
                            {'label': 'Custom (Manual Entry)', 'value': 'custom'},
                            {'label': 'End of Each Month', 'value': 'end_of_month'},
                            {'label': 'Beginning of Each Month', 'value': 'beginning_of_month'},
                            {'label': 'Middle of Each Month (15th)', 'value': 'middle_of_month'},
                            {'label': 'Every Friday', 'value': 'every_friday'},
                            {'label': 'Every Monday', 'value': 'every_monday'},
                            {'label': 'First Friday of Each Month', 'value': 'first_friday'},
                            {'label': 'Last Friday of Each Month', 'value': 'last_friday'},
                            {'label': 'Every Other Week (Mondays)', 'value': 'biweekly_monday'},
                            {'label': 'Quarterly (End of Quarter)', 'value': 'quarterly'}
                        ],
                        value='custom',
                        className="mb-3",
                        style={'color': 'black'}
                    ),
                    
                    html.Label("Specific Dates (one per line, YYYY-MM-DD):"),
                    dcc.Textarea(
                        id='specific-dates',
                        value='''2025-01-31
2025-02-28
2025-03-31
2025-04-30
2025-05-31
2025-06-30
2025-07-31
2025-08-31''',
                        style={'width': '100%', 'height': 200},
                        className="form-control mb-3"
                    ),
                    
                    dbc.Button("Generate Analysis", id="generate-btn", color="primary", className="w-100")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    html.Div(id='output-message', className="mb-3"),
                    dcc.Graph(id='yield-change-plot'),
                    dcc.Graph(id='sharpe-plot'),
                    dcc.Graph(id='hit-rate-plot'),
                    dcc.Graph(id='time-series-plot'),
                    html.Div(id='tables-container', className="mt-4")
                ]
            )
        ], width=9)
    ])
], fluid=True)


def generate_plots(days_window, tenor, specific_dates_str):
    """Generate all plots as Plotly figures"""
    
    # Parse dates
    specific_dates = [date.strip() for date in specific_dates_str.replace('\n', ',').split(',') if date.strip()]
    
    # Map tenor to FRED series ID
    tenor_map = {
        '2Y': 'DGS2', '3Y': 'DGS3', '5Y': 'DGS5', '7Y': 'DGS7',
        '10Y': 'DGS10', '20Y': 'DGS20', '30Y': 'DGS30'
    }
    
    tenor_upper = tenor.upper()
    series_id = tenor_map[tenor_upper]
    
    # Calculate date range
    specific_dates_dt = pd.to_datetime(specific_dates)
    min_date = specific_dates_dt.min() - pd.Timedelta(days=days_window + 10)
    max_date = specific_dates_dt.max() + pd.Timedelta(days=days_window + 10)
    
    # Fetch data
    dataframe = pdr.DataReader(series_id, 'fred', min_date, max_date)
    df_copy = pd.DataFrame({
        'observation_date': dataframe.index,
        series_id: dataframe[series_id].values
    })
    df_copy['observation_date'] = pd.to_datetime(df_copy['observation_date'])
    yield_col = series_id
    df_copy = df_copy.dropna(subset=[yield_col]).reset_index(drop=True)
    
    # Filter for specific dates
    df_filtered = df_copy[df_copy['observation_date'].isin(specific_dates_dt)].reset_index(drop=True)
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
    change_matrix = np.divide(change_matrix, count_matrix, where=count_matrix > 0, out=np.full_like(change_matrix, np.nan))
    
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
    
    # Create axis labels
    axis_labels = [str(i) for i in range(-days_window, days_window + 1)]
    
    # Create text arrays with empty strings for NaN values
    text_change = np.where(np.isnan(change_matrix), '', np.round(change_matrix * 100, 1).astype(str))
    text_sharpe = np.where(np.isnan(sharpe_matrix), '', np.round(sharpe_matrix, 2).astype(str))
    text_hit_rate = np.where(np.isnan(hit_rate_matrix), '', np.round(hit_rate_matrix * 100, 0).astype(str))
    
    # 1. Yield Change Heatmap
    fig1 = go.Figure(data=go.Heatmap(
        z=change_matrix * 100,
        x=axis_labels,
        y=axis_labels,
        colorscale='RdYlGn',
        colorbar=dict(title='Yield Change (bps)'),
        text=text_change,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},
        hovertemplate='Start: %{y}<br>End: %{x}<br>Change: %{z:.1f} bps<extra></extra>'
    ))
    fig1.update_layout(
        title=f'UST {tenor_upper} Yield Changes Heatmap<br>{len(specific_dates)} specific dates',
        xaxis_title='End Days Offset',
        yaxis_title='Start Days Offset',
        yaxis=dict(autorange='reversed'),
        template='plotly_dark',
        height=700
    )
    
    # 2. Sharpe Ratio Heatmap
    fig2 = go.Figure(data=go.Heatmap(
        z=sharpe_matrix,
        x=axis_labels,
        y=axis_labels,
        colorscale='RdYlGn',
        colorbar=dict(title='Sharpe Ratio'),
        text=text_sharpe,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},
        hovertemplate='Start: %{y}<br>End: %{x}<br>Sharpe: %{z:.2f}<extra></extra>'
    ))
    fig2.update_layout(
        title=f'UST {tenor_upper} Sharpe Ratio Heatmap<br>{len(specific_dates)} specific dates',
        xaxis_title='End Days Offset',
        yaxis_title='Start Days Offset',
        yaxis=dict(autorange='reversed'),
        template='plotly_dark',
        height=700
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
        title=f'UST {tenor_upper} Hit Rate Heatmap<br>{len(specific_dates)} specific dates',
        xaxis_title='End Days Offset',
        yaxis_title='Start Days Offset',
        yaxis=dict(autorange='reversed'),
        template='plotly_dark',
        height=700
    )
    
    # 4. Normalized Time Series
    fig4 = go.Figure()
    for current_date, (offsets, normalized_yields) in normalized_series.items():
        fig4.add_trace(go.Scatter(
            x=offsets,
            y=[y * 100 for y in normalized_yields],
            mode='lines+markers',
            name=current_date.strftime('%Y-%m-%d'),
            hovertemplate='%{x} days<br>%{y:.1f} bps<extra></extra>'
        ))
    
    fig4.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig4.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    fig4.update_layout(
        title=f'Normalized UST {tenor_upper} Yield Time Series<br>{len(specific_dates)} specific dates',
        xaxis_title='Days Offset from Reference Date',
        yaxis_title='Yield Change (basis points)',
        template='plotly_dark',
        height=600,
        hovermode='closest'
    )
    
    # Calculate statistics for tables
    # Find best and worst Sharpe trades
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
                    'trade_details': trade_details[key]  # Include detailed trade information
                })

    valid_sharpes.sort(key=lambda x: x['sharpe'], reverse=True)

    table_data = {
        'best_trade': valid_sharpes[0] if valid_sharpes else None,
        'worst_trade': valid_sharpes[-1] if valid_sharpes else None,
        'overall_stats': valid_sharpes
    }
    
    return fig1, fig2, fig3, fig4, table_data


def generate_date_preset(preset):
    """Generate dates based on the selected preset for 2025"""
    from datetime import datetime, timedelta
    import calendar
    
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
        # Find first Friday
        while current.weekday() != 4:  # 4 = Friday
            current += timedelta(days=1)
        while current.year == 2025:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
    
    elif preset == 'every_monday':
        # Every Monday in 2025
        current = datetime(2025, 1, 1)
        # Find first Monday
        while current.weekday() != 0:  # 0 = Monday
            current += timedelta(days=1)
        while current.year == 2025:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
    
    elif preset == 'first_friday':
        # First Friday of each month
        for month in range(1, 13):
            current = datetime(2025, month, 1)
            # Find first Friday
            while current.weekday() != 4:
                current += timedelta(days=1)
            dates.append(current.strftime('%Y-%m-%d'))
    
    elif preset == 'last_friday':
        # Last Friday of each month
        for month in range(1, 13):
            last_day = calendar.monthrange(2025, month)[1]
            current = datetime(2025, month, last_day)
            # Find last Friday
            while current.weekday() != 4:
                current -= timedelta(days=1)
            dates.append(current.strftime('%Y-%m-%d'))
    
    elif preset == 'biweekly_monday':
        # Every other Monday starting from first Monday
        current = datetime(2025, 1, 1)
        # Find first Monday
        while current.weekday() != 0:
            current += timedelta(days=1)
        while current.year == 2025:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=14)
    
    elif preset == 'quarterly':
        # End of each quarter (Mar, Jun, Sep, Dec)
        dates = ['2025-03-31', '2025-06-30', '2025-09-30', '2025-12-31']
    
    return '\n'.join(dates)


@app.callback(
    Output('specific-dates', 'value'),
    Input('date-preset', 'value')
)
def update_dates_from_preset(preset):
    """Update the dates textarea when a preset is selected"""
    if preset:
        return generate_date_preset(preset)
    return ''


@app.callback(
    [Output('yield-change-plot', 'figure'),
     Output('sharpe-plot', 'figure'),
     Output('hit-rate-plot', 'figure'),
     Output('time-series-plot', 'figure'),
     Output('tables-container', 'children'),
     Output('output-message', 'children')],
    [Input('generate-btn', 'n_clicks')],
    [State('days-window', 'value'),
     State('tenor', 'value'),
     State('specific-dates', 'value')]
)
def update_plots(n_clicks, days_window, tenor, specific_dates):
    if n_clicks is None:
        return [{}, {}, {}, {}, html.Div(), "Set parameters and click 'Generate Analysis'"]
    
    try:
        fig1, fig2, fig3, fig4, table_data = generate_plots(days_window, tenor, specific_dates)
        
        # Create tables
        tables = []
        
        # Best Sharpe Trade Table
        if table_data['best_trade']:
            best = table_data['best_trade']
            
            # Summary table
            summary_rows = [
                html.Tr([html.Td("Start Offset"), html.Td(f"{best['start_offset']} days")]),
                html.Tr([html.Td("End Offset"), html.Td(f"{best['end_offset']} days")]),
                html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{best['sharpe']:.2f}")]),
                html.Tr([html.Td("Mean Change"), html.Td(f"{best['mean_change']:.1f} bps")]),
                html.Tr([html.Td("Hit Rate"), html.Td(f"{best['hit_rate']:.1f}%")]),
                html.Tr([html.Td("Number of Trades"), html.Td(f"{best['num_trades']}")])
            ]
            
            # Detailed trades table
            detail_header = html.Tr([
                html.Th("Reference Date"),
                html.Th("Start Date"),
                html.Th("Start Yield"),
                html.Th("End Date"),
                html.Th("End Yield"),
                html.Th("Change (bps)")
            ])
            
            detail_rows = []
            for trade in best['trade_details']:
                detail_rows.append(html.Tr([
                    html.Td(trade['reference_date'].strftime('%Y-%m-%d')),
                    html.Td(trade['start_date'].strftime('%Y-%m-%d')),
                    html.Td(f"{trade['start_yield']:.3f}%"),
                    html.Td(trade['end_date'].strftime('%Y-%m-%d')),
                    html.Td(f"{trade['end_yield']:.3f}%"),
                    html.Td(f"{trade['change']*100:.1f}")
                ]))
            
            tables.append(dbc.Card([
                dbc.CardHeader(html.H5("Best Sharpe Ratio Trade")),
                dbc.CardBody([
                    html.H6("Summary", className="mb-2"),
                    dbc.Table([html.Tbody(summary_rows)], bordered=True, striped=True, hover=True, className="mb-3"),
                    html.H6("Trade Details", className="mb-2"),
                    dbc.Table([
                        html.Thead([detail_header]),
                        html.Tbody(detail_rows)
                    ], bordered=True, striped=True, hover=True, responsive=True)
                ])
            ], className="mb-3"))
        
        # Worst Sharpe Trade Table
        if table_data['worst_trade']:
            worst = table_data['worst_trade']
            
            # Summary table
            summary_rows = [
                html.Tr([html.Td("Start Offset"), html.Td(f"{worst['start_offset']} days")]),
                html.Tr([html.Td("End Offset"), html.Td(f"{worst['end_offset']} days")]),
                html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{worst['sharpe']:.2f}")]),
                html.Tr([html.Td("Mean Change"), html.Td(f"{worst['mean_change']:.1f} bps")]),
                html.Tr([html.Td("Hit Rate"), html.Td(f"{worst['hit_rate']:.1f}%")]),
                html.Tr([html.Td("Number of Trades"), html.Td(f"{worst['num_trades']}")])
            ]
            
            # Detailed trades table
            detail_header = html.Tr([
                html.Th("Reference Date"),
                html.Th("Start Date"),
                html.Th("Start Yield"),
                html.Th("End Date"),
                html.Th("End Yield"),
                html.Th("Change (bps)")
            ])
            
            detail_rows = []
            for trade in worst['trade_details']:
                detail_rows.append(html.Tr([
                    html.Td(trade['reference_date'].strftime('%Y-%m-%d')),
                    html.Td(trade['start_date'].strftime('%Y-%m-%d')),
                    html.Td(f"{trade['start_yield']:.3f}%"),
                    html.Td(trade['end_date'].strftime('%Y-%m-%d')),
                    html.Td(f"{trade['end_yield']:.3f}%"),
                    html.Td(f"{trade['change']*100:.1f}")
                ]))
            
            tables.append(dbc.Card([
                dbc.CardHeader(html.H5("Worst Sharpe Ratio Trade")),
                dbc.CardBody([
                    html.H6("Summary", className="mb-2"),
                    dbc.Table([html.Tbody(summary_rows)], bordered=True, striped=True, hover=True, className="mb-3"),
                    html.H6("Trade Details", className="mb-2"),
                    dbc.Table([
                        html.Thead([detail_header]),
                        html.Tbody(detail_rows)
                    ], bordered=True, striped=True, hover=True, responsive=True)
                ])
            ], className="mb-3"))
        
        # Overall Statistics Table
        if table_data['overall_stats']:
            all_sharpes = [x['sharpe'] for x in table_data['overall_stats']]
            all_changes = [x['mean_change'] for x in table_data['overall_stats']]
            all_hit_rates = [x['hit_rate'] for x in table_data['overall_stats']]
            
            tables.append(dbc.Card([
                dbc.CardHeader(html.H5("Overall Statistics")),
                dbc.CardBody([
                    dbc.Table([
                        html.Tbody([
                            html.Tr([html.Td("Total Trade Combinations"), html.Td(f"{len(table_data['overall_stats'])}")]),
                            html.Tr([html.Td("Average Sharpe Ratio"), html.Td(f"{np.mean(all_sharpes):.2f}")]),
                            html.Tr([html.Td("Median Sharpe Ratio"), html.Td(f"{np.median(all_sharpes):.2f}")]),
                            html.Tr([html.Td("Average Mean Change"), html.Td(f"{np.mean(all_changes):.1f} bps")]),
                            html.Tr([html.Td("Average Hit Rate"), html.Td(f"{np.mean(all_hit_rates):.1f}%")]),
                            html.Tr([html.Td("Positive Sharpe Trades"), html.Td(f"{sum(1 for s in all_sharpes if s > 0)} ({sum(1 for s in all_sharpes if s > 0)/len(all_sharpes)*100:.1f}%)")])
                        ])
                    ], bordered=True, striped=True, hover=True)
                ])
            ], className="mb-3"))
        
        tables_container = html.Div(tables)
        
        return [fig1, fig2, fig3, fig4, tables_container, dbc.Alert("Analysis generated successfully!", color="success")]
    except Exception as e:
        return [{}, {}, {}, {}, html.Div(), dbc.Alert(f"Error: {str(e)}", color="danger")]


if __name__ == '__main__':
    app.run(debug=True)
