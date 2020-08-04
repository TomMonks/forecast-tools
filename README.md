# forecast-tools

[![DOI](https://zenodo.org/badge/250494795.svg)](https://zenodo.org/badge/latestdoi/250494795)
[![PyPI version fury.io](https://badge.fury.io/py/forecast-tools.svg)](https://pypi.python.org/pypi/forecast-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/forecast-tools/master)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)

 **Fundamental tools to support the forecasting process in python.**

 forecast-tools has been developed to support forecasting education and applied forecasting research.  It is MIT licensed and freely available to practitioners, students and researchers via PyPi.  There is a long term plan to make forecast-tools available via conda-forge.

## Two ways to checkout forecast-tools

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

Unit tests are provided and can be run from the command `pytest`.   All contributions are welcome and must include unit tests!