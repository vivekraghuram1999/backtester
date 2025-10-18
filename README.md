# FI RV Strategy Backtester

A Python-based backtesting tool for analyzing Fixed Income (FI) Relative Value (RV) trading strategies on US Treasury yields. This interactive dashboard allows users to visualize yield changes, Sharpe ratios, and hit rates across different time windows and specific dates.

## Features

- **Interactive Dashboard**: Built with Dash and Plotly for real-time data visualization
- **Multiple Treasury Tenors**: Support for 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, and 30Y US Treasuries
- **Flexible Date Selection**: Choose from preset date patterns or specify custom dates
- **Comprehensive Analytics**:
  - Yield change heatmaps showing basis point changes
  - Sharpe ratio calculations for risk-adjusted returns
  - Hit rate analysis for strategy success rates
  - Normalized time series visualization
  - Detailed trade breakdown tables

## Date Presets

The backtester includes several convenient date presets:
- End of Each Month
- Beginning of Each Month
- Middle of Each Month (15th)
- Every Friday / Monday
- First/Last Friday of Each Month
- Every Other Week (Mondays)
- Quarterly (End of Quarter)
- Custom (Manual Entry)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vivekraghuram1999/backtester.git
cd backtester
```

2. Install required dependencies:
```bash
pip install dash dash-bootstrap-components pandas numpy pandas-datareader plotly
```

### Required Packages
- `dash` - Web application framework
- `dash-bootstrap-components` - Bootstrap components for Dash
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `pandas-datareader` - Data access from FRED API
- `plotly` - Interactive plotting library

## Usage

### Running the Dashboard

1. Start the Dash application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

3. Configure your backtest parameters:
   - **Days Window**: Number of days before and after each reference date (1-30)
   - **Treasury Tenor**: Select from 2Y to 30Y treasuries
   - **Date Preset**: Choose a preset pattern or select "Custom" for manual entry
   - **Specific Dates**: Enter dates in YYYY-MM-DD format (one per line) if using custom dates

4. Click "Generate Analysis" to run the backtest

### Using the Jupyter Notebook

The repository includes a Jupyter notebook (`backtester.ipynb`) for exploratory analysis:

```bash
jupyter notebook backtester.ipynb
```

## Data Source

Treasury yield data is fetched from the Federal Reserve Economic Data (FRED) API using `pandas_datareader`. The following FRED series are used:

| Tenor | FRED Series ID |
|-------|----------------|
| 2Y    | DGS2           |
| 3Y    | DGS3           |
| 5Y    | DGS5           |
| 7Y    | DGS7           |
| 10Y   | DGS10          |
| 20Y   | DGS20          |
| 30Y   | DGS30          |

## Output

The backtester generates four main visualizations:

1. **Yield Changes Heatmap**: Shows average yield changes (in basis points) for different entry/exit day combinations
2. **Sharpe Ratio Heatmap**: Displays risk-adjusted returns for each trading window
3. **Hit Rate Heatmap**: Shows the percentage of profitable trades for each strategy
4. **Normalized Time Series Plot**: Visualizes yield movements relative to each reference date

Additionally, detailed tables provide:
- Best performing strategies ranked by Sharpe ratio
- Individual trade breakdowns with dates and yield changes

## Example

For a backtest with:
- Days Window: 7
- Tenor: 10Y
- Dates: End of each month in 2025

The tool will analyze how 10-year Treasury yields behaved in the 7 days before and after each month-end, helping identify consistent patterns in yield movements.

## Project Structure

```
backtester/
├── app.py              # Main Dash application (UI and callbacks)
├── data_analysis.py    # Data parsing and analysis logic
├── backtester.ipynb    # Jupyter notebook for analysis
├── DGS10.csv          # Sample 10-year Treasury data
└── README.md          # This file
```

## Code Organization

The project is organized into two main Python files:

### `app.py` - Application Structure
- Dash application initialization
- UI layout and components (inputs, dropdowns, graphs)
- Callback functions for user interactions
- Table generation for displaying results

### `data_analysis.py` - Data Processing
- Data fetching from FRED API
- Matrix calculations (yield changes, Sharpe ratios, hit rates)
- Plot generation functions
- Date preset generation
- Statistical analysis utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Data provided by the Federal Reserve Bank of St. Louis (FRED)
- Built with Dash by Plotly
