# forecast-tools: fundamental tools to support the forecasting process in python.

[![DOI](https://zenodo.org/badge/250494795.svg)](https://zenodo.org/badge/latestdoi/250494795)
[![PyPI version fury.io](https://badge.fury.io/py/forecast-tools.svg)](https://pypi.python.org/pypi/forecast-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/forecast-tools/master)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)

 forecast-tools has been developed to support forecasting education and applied forecasting research.  It is MIT licensed and freely available to practitioners, students and researchers via PyPi.  There is a long term plan to make forecast-tools available via conda-forge.

 # Vision for forecast-tools

 1. Deliverhigh quality reliable code for forecasting education and practice with full documentation and unit testing.
 2. Provide a simple to use pythonic interface that users of `statsmodels` and `sklearn` will recognise.
 3. To improve the quality of Machine Learning time series forecasting and encourage the use of best practice.

# Features:

1. Implementation of classic naive forecast benchmarks such as Naive Forecast 1 along with prediction intervals
2. Implementation of scale-dependent, relative and scaled forecast errors.
3. Rolling forecast origin and sliding window for time series cross validation
4. Built in daily level datasets

## Two simple ways to explore forecast-tools

1. `pip install forecast-tools`
2. Click on the launch-binder at the top of this readme. This will open example Jupyter notebooks in the cloud via Binder.

## Citation

If you use forecast-tools for research, a practical report, education or any reason please include the following citation.

> Monks, Thomas. (2020). forecast-tools: fundamental tools to support the forecasting process in python. Zenodo. http://doi.org/10.5281/zenodo.3969789

```tex
@software{forecast_tools_3969789,
  author       = {Thomas Monks},
  title        = {forecast-tools: fundamental tools to support the forecasting process in python},
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3969789},
  url          = {https://doi.org/10.5281/zenodo.3969789}
}
```

## Contributing to forecast-tools

Please fork Dev, make your modifications, run the unit tests and submit a pull request for review.

Development environment:

* `conda env create -f binder/environment.yml`

* `conda activate forecast_dev`

Unit tests are provided and can be run from the command `pytest` and its coverage extension.  Run the following in the terminal.

* `pytest --cov=forecast_tools tests/`

**All contributions are welcome and must include unit tests!**