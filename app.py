"""
Dash application for FI RV Strategy Backtester.
Main application structure and UI components.
"""

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
from data_analysis import generate_plots, generate_date_preset

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
