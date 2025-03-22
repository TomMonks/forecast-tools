import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from forecast_tools.plotting import plot_time_series  

# Fixtures for common test data
@pytest.fixture
def sample_dates():
    """Create a sample DatetimeIndex."""
    return pd.date_range(start='2023-01-01', periods=10, freq='D')

@pytest.fixture
def training_data(sample_dates):
    """Create sample training data."""
    return pd.DataFrame({'value': np.sin(np.arange(10)) + 10}, index=sample_dates)

@pytest.fixture
def test_data(sample_dates):
    """Create sample test data."""
    test_dates = pd.date_range(start='2023-01-11', periods=5, freq='D')
    return pd.DataFrame({'value': np.sin(np.arange(10, 15)) + 10}, index=test_dates)

@pytest.fixture
def forecast_data(sample_dates):
    """Create sample forecast data."""
    forecast_dates = pd.date_range(start='2023-01-11', periods=5, freq='D')
    return pd.DataFrame({'value': np.sin(np.arange(10, 15)) + 10.5}, index=forecast_dates)

@pytest.fixture
def prediction_intervals(sample_dates):
    """Create sample prediction intervals."""
    forecast_dates = pd.date_range(start='2023-01-11', periods=5, freq='D')
    base_values = np.sin(np.arange(10, 15)) + 10.5
    
    intervals = {}
    for width in [95, 80, 50]:
        half_width = width / 100 * 2
        intervals[f"{width}% PI"] = pd.DataFrame({
            'lower': base_values - half_width,
            'upper': base_values + half_width
        }, index=forecast_dates)
    
    return intervals

# Basic functionality tests
def test_basic_plot(training_data):
    """Test that the function works with just training data."""
    fig = plot_time_series(training_data, show_figure=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Only training data trace

def test_with_test_data(training_data, test_data):
    """Test with training and test data."""
    fig = plot_time_series(training_data, test_data, show_figure=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Training and test data traces

def test_with_forecast(training_data, forecast_data):
    """Test with training and forecast data."""
    fig = plot_time_series(training_data, forecast=forecast_data, show_figure=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Training and forecast traces

def test_complete_plot(training_data, test_data, forecast_data, prediction_intervals):
    """Test with all components."""
    fig = plot_time_series(
        training_data, 
        test_data, 
        forecast_data, 
        prediction_intervals,
        show_figure=False
    )
    assert isinstance(fig, go.Figure)
    # Training + test + forecast + 2 traces per interval (3 intervals)
    assert len(fig.data) == 1 + 1 + 1 + (2 * 3)

# Edge cases and error handling tests
def test_empty_training_data():
    """Test with empty training data."""
    empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        plot_time_series(empty_df, show_figure=False)

def test_non_dataframe_input():
    """Test with non-DataFrame input."""
    with pytest.raises(TypeError):
        plot_time_series([1, 2, 3], show_figure=False)

def test_non_datetime_index():
    """Test with non-DatetimeIndex."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    with pytest.raises(TypeError):
        plot_time_series(df, show_figure=False)

def test_multiple_columns():
    """Test with multiple columns in training data."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'value1': np.random.rand(10),
        'value2': np.random.rand(10)
    }, index=dates)
    with pytest.raises(ValueError):
        plot_time_series(df, show_figure=False)

def test_invalid_test_data_mode():
    """Test with invalid test_data_mode."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({'value': np.random.rand(10)}, index=dates)
    with pytest.raises(ValueError):
        plot_time_series(df, test_data_mode="invalid_mode", show_figure=False)

def test_invalid_prediction_intervals():
    """Test with invalid prediction intervals."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    train_df = pd.DataFrame({'value': np.random.rand(10)}, index=dates)
    
    # Missing required columns
    invalid_pi = {
        "95% PI": pd.DataFrame({'wrong_column': np.random.rand(5)}, 
                              index=pd.date_range(start='2023-01-11', periods=5, freq='D'))
    }
    
    with pytest.raises(ValueError):
        plot_time_series(train_df, prediction_intervals=invalid_pi, show_figure=False)

def test_custom_colors(training_data):
    """Test with custom colors."""
    custom_colors = {
        "training": "#FF5733",
        "test": "#33FF57",
        "forecast": "#3357FF"
    }
    fig = plot_time_series(training_data, custom_colors=custom_colors, show_figure=False)
    assert fig.data[0].line.color == "#FF5733"

def test_different_color_schemes(training_data):
    """Test different color schemes."""
    for scheme in ["red", "blue", "green"]:
        fig = plot_time_series(training_data, color_scheme=scheme, show_figure=False)
        assert isinstance(fig, go.Figure)

def test_forecast_line_styles(training_data, forecast_data):
    """Test different forecast line styles."""
    for style in ["dash", "solid", "dot", "dashdot"]:
        fig = plot_time_series(
            training_data, 
            forecast=forecast_data, 
            forecast_line_style=style,
            show_figure=False
        )
        assert isinstance(fig, go.Figure)
