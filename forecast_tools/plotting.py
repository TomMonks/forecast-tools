"""
Time Series Visualization Module
===============================

This module provides tools for visualizing time series data, forecasts, and prediction
intervals using Plotly. It is designed to help students learn about forecasting.

The main function, `plot_time_series`, creates interactive visualizations of univariate
time series data with options to display training data, test data, point forecasts,
and prediction intervals at various confidence levels.

Features
--------
- Interactive Plotly-based visualizations
- Support for displaying training and test data
- Visualization of point forecasts
- Display of prediction intervals at multiple confidence levels
- Customizable colors and styling
- Hover functionality with vertical guide lines
- Multiple predefined color schemes

Example Usage
------------
>>> import pandas as pd
>>> from forecast_tools.plotting import plot_time_series
>>>
>>> # Basic usage with training data only
>>> plot_time_series(training_data=train_df)
>>>
>>> # With training, test data, and a forecast
>>> plot_time_series(
>>>     training_data=train_df,
>>>     test_data=test_df,
>>>     forecast=forecast_df
>>> )
>>>
>>> # With prediction intervals
>>> intervals = {
>>>     "95% PI": pi_95_df,
>>>     "80% PI": pi_80_df
>>> }
>>> plot_time_series(
>>>     training_data=train_df,
>>>     forecast=forecast_df,
>>>     prediction_intervals=intervals,
>>>     color_scheme="blue"
>>> )
>>>
>>> # Save the figure for later use
>>> fig = plot_time_series(
>>>     training_data=train_df,
>>>     test_data=test_df,
>>>     show_figure=False
>>> )
>>> fig.write_html("my_time_series.html")

Notes
-----
This module is intended for educational purposes to help students understand
forecasting concepts through visualization. The interactive nature of the plots
allows for exploration of time series patterns, forecast accuracy, and uncertainty.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Literal

# Color scheme dictionaries
COLOR_SCHEMES = {
    "red": {
        "95% PI": "rgba(255, 0, 0, 0.15)",  # Lightest red
        "90% PI": "rgba(255, 0, 0, 0.18)",
        "80% PI": "rgba(255, 0, 0, 0.21)",
        "70% PI": "rgba(255, 0, 0, 0.24)",
        "60% PI": "rgba(255, 0, 0, 0.27)",
        "50% PI": "rgba(255, 0, 0, 0.30)",
        "40% PI": "rgba(255, 0, 0, 0.33)",
        "30% PI": "rgba(255, 0, 0, 0.36)",
        "20% PI": "rgba(255, 0, 0, 0.39)",
        "10% PI": "rgba(255, 0, 0, 0.42)",  # Darkest red
        "forecast": "#FF0000",
        "training": "#0072B2",
        "test": "#000000",
    },
    "blue": {
        "95% PI": "rgba(0, 0, 255, 0.15)",  # Lightest blue
        "90% PI": "rgba(0, 0, 255, 0.18)",
        "80% PI": "rgba(0, 0, 255, 0.21)",
        "70% PI": "rgba(0, 0, 255, 0.24)",
        "60% PI": "rgba(0, 0, 255, 0.27)",
        "50% PI": "rgba(0, 0, 255, 0.30)",
        "40% PI": "rgba(0, 0, 255, 0.33)",
        "30% PI": "rgba(0, 0, 255, 0.36)",
        "20% PI": "rgba(0, 0, 255, 0.39)",
        "10% PI": "rgba(0, 0, 255, 0.42)",  # Darkest blue
        "forecast": "#0000FF",
        "training": "#FF9933",
        "test": "#000000",
    },
    "green": {
        "95% PI": "rgba(0, 128, 0, 0.15)",  # Lightest green
        "90% PI": "rgba(0, 128, 0, 0.18)",
        "80% PI": "rgba(0, 128, 0, 0.21)",
        "70% PI": "rgba(0, 128, 0, 0.24)",
        "60% PI": "rgba(0, 128, 0, 0.27)",
        "50% PI": "rgba(0, 128, 0, 0.30)",
        "40% PI": "rgba(0, 128, 0, 0.33)",
        "30% PI": "rgba(0, 128, 0, 0.36)",
        "20% PI": "rgba(0, 128, 0, 0.39)",
        "10% PI": "rgba(0, 128, 0, 0.42)",  # Darkest green
        "forecast": "#008000",
        "training": "#9966FF",
        "test": "#000000",
    },
}


def plot_time_series(
    training_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    forecast: Optional[pd.DataFrame] = None,
    prediction_intervals: Optional[Dict[str, pd.DataFrame]] = None,
    test_data_mode: Literal["markers", "lines"] = "markers",
    y_axis_label: str = "Value",
    custom_colors: Optional[Dict[str, str]] = None,
    color_scheme: Literal["red", "blue", "green"] = "red",
    include_title: bool = True,
    forecast_line_style: Literal["dash", "solid", "dot", "dashdot"] = "dash",
    figure_width: int = 900,
    figure_height: int = 500,
    show_figure: bool = True,
) -> go.Figure:
    """
    Plots univariate time series data using Plotly. Options for displaying
    training data, test data, forecasts, and prediction intervals.

    Parameters:
    ----------
    training_data: pd.DataFrame
        A pandas DataFrame with a DatetimeIndex containing the training data.
        It should have one column representing the time series values.

    test_data: pd.DataFrame, optional (default=None)
        A pandas DataFrame with a DatetimeIndex containing the test data.
        Displayed as black dots or a line based on `test_data_mode`.

    forecast: pd.DataFrame, optional (default=None)
        A pandas DataFrame with a DatetimeIndex containing the forecasted data.
        Should have one column representing the point forecast values.

    prediction_intervals: dict[str, pd.DataFrame], Optional (default=None):
        A dictionary of pandas dataframes that contain prediction intervals.
        Each dataframe must contain two columns ('upper' and 'lower') and a
        DatetimeIndex (over the forecast period). The key for the dictionary
        should follow the convention '95% PI', '80% PI', etc.

    test_data_mode: str, optional (default='markers')
        Mode for displaying test data.
        Must be either 'markers' (dots) or 'lines' (line).

    y_axis_label: str, optional (default='Value')
        The quantity measured by the time series to display on the y-axis.

    custom_colors: dict, optional (default=None)
        Dictionary with color codes for 'training', 'test', 'forecast',
        and prediction intervals. Overrides the color_scheme.

    color_scheme: str, optional (default='red')
        Predefined color scheme to use. Options: 'red', 'blue', 'green'.

    include_title: bool, optional (default=True)
        Whether to include a default title on the plot.

    forecast_line_style: str, optional (default='dash')
        Line style for the forecast. Options: 'dash', 'solid', 'dot', 'dashdot'.

    figure_width: int, optional (default=900)
        Width of the figure in pixels.

    figure_height: int, optional (default=500)
        Height of the figure in pixels.

    show_figure: bool, optional (default=True)
        Whether to display the figure immediately.

    Returns:
        go.Figure: A Plotly figure object that can be further customized or displayed.

    Examples:
    ---------
    >>> # Basic usage with training data only
    >>> plot_time_series(training_data=df)
    >>>
    >>> # With training, test data, and a forecast
    >>> plot_time_series(
    >>>     training_data=train_df,
    >>>     test_data=test_df,
    >>>     forecast=forecast_df
    >>> )
    >>>
    >>> # With prediction intervals
    >>> intervals = {
    >>>     "95% PI": pi_95_df,
    >>>     "80% PI": pi_80_df
    >>> }
    >>> plot_time_series(
    >>>     training_data=train_df,
    >>>     forecast=forecast_df,
    >>>     prediction_intervals=intervals,
    >>>     color_scheme="blue"
    >>> )
    """
    # Validate inputs
    if not isinstance(training_data, pd.DataFrame):
        raise TypeError("training_data must be a pandas DataFrame")

    if not isinstance(training_data.index, pd.DatetimeIndex):
        raise TypeError("training_data must have a DatetimeIndex")

    if training_data.shape[1] != 1:
        raise ValueError("training_data should have exactly one column")

    if test_data is not None:
        if not isinstance(test_data, pd.DataFrame):
            raise TypeError("test_data must be a pandas DataFrame")
        if not isinstance(test_data.index, pd.DatetimeIndex):
            raise TypeError("test_data must have a DatetimeIndex")
        if test_data.shape[1] != 1:
            raise ValueError("test_data should have exactly one column")

    if forecast is not None:
        if not isinstance(forecast, pd.DataFrame):
            raise TypeError("forecast must be a pandas DataFrame")
        if not isinstance(forecast.index, pd.DatetimeIndex):
            raise TypeError("forecast must have a DatetimeIndex")
        if forecast.shape[1] != 1:
            raise ValueError("forecast should have exactly one column")

    # Validate test_data_mode input
    if test_data_mode not in ["markers", "lines"]:
        raise ValueError(
            "Invalid value for test_data_mode. Choose 'markers' or 'lines'."
        )

    # Validate forecast_line_style
    if forecast_line_style not in ["dash", "solid", "dot", "dashdot"]:
        raise ValueError(
            "Invalid value for forecast_line_style. Choose 'dash', 'solid', 'dot', or 'dashdot'."
        )

    # Validate color_scheme
    if color_scheme not in COLOR_SCHEMES:
        raise ValueError(
            f"Invalid color_scheme. Choose from: {', '.join(COLOR_SCHEMES.keys())}"
        )

    # Set up colors
    colors = COLOR_SCHEMES[color_scheme].copy()

    # Override with custom colors if provided
    if custom_colors is not None:
        colors.update(custom_colors)

    # Create a Plotly figure
    fig = go.Figure()

    # Add training data as a line plot
    fig.add_trace(
        go.Scatter(
            x=training_data.index,
            y=training_data.iloc[:, 0],
            mode="lines",
            name="Training Data",
            line=dict(color=colors["training"]),
        )
    )

    # Add prediction intervals before forecast
    if prediction_intervals is not None:
        # Sort intervals by width (assuming naming convention like "95% PI")
        sorted_intervals = sorted(
            prediction_intervals.items(),
            key=lambda x: float(x[0].split("%")[0]),
            reverse=True,
        )

        for interval_name, interval_df in sorted_intervals:
            # Validate interval dataframe
            if not isinstance(interval_df, pd.DataFrame):
                raise TypeError(f"{interval_name} must be a pandas DataFrame")

            if not isinstance(interval_df.index, pd.DatetimeIndex):
                raise TypeError(f"{interval_name} must have a DatetimeIndex")

            if not {"lower", "upper"}.issubset(interval_df.columns):
                raise ValueError(
                    f"{interval_name} DataFrame must contain 'lower' and 'upper' columns"
                )

            interval_color = colors.get(
                interval_name, colors.get("95% PI", "rgba(255, 0, 0, 0.15)")
            )

            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=interval_df.index,
                    y=interval_df["upper"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    name=f"{interval_name} (Upper)",
                )
            )

            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=interval_df.index,
                    y=interval_df["lower"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=interval_color,
                    name=f"{interval_name}",
                    showlegend=True,
                )
            )

    # Add forecast line
    if forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.iloc[:, 0],
                mode="lines",
                name="Point Forecast",
                line=dict(color=colors["forecast"], dash=forecast_line_style),
            )
        )

    # Add test data based on the selected mode
    if test_data is not None:
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_data.iloc[:, 0],
                mode=test_data_mode,
                name="Test Data",
                marker=(
                    dict(color=colors["test"], size=6)
                    if test_data_mode == "markers"
                    else None
                ),
                line=(
                    dict(color=colors["test"]) if test_data_mode == "lines" else None
                ),
            )
        )

    title = "Univariate Time Series Visualization" if include_title else ""

    # Enable vertical spike lines on hover
    fig.update_layout(
        title=title,
        width=figure_width,
        height=figure_height,
        xaxis=dict(
            showspikes=True,  # Enable spikes
            spikemode="across",  # Spike spans across all traces
            spikesnap="cursor",  # Spike snaps to cursor position
            spikedash="dot",  # Style of spike line
            spikethickness=1.5,  # Thickness of spike line
            spikecolor="gray",  # Color of spike line
        ),
        yaxis=dict(showspikes=False),  # Disable horizontal spikes
        hovermode="x unified",  # Show all points at same x-position
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=y_axis_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Show the figure if requested
    if show_figure:
        fig.show()

    # Return the figure object for further customization
    return fig
