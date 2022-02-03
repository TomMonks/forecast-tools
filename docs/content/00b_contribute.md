# Contributing

Please fork Dev, make your modifications, run the unit tests and submit a pull request to dev for review.

Development environment:

* `conda env create -f binder/environment.yml`

* `conda activate forecast_dev`

Unit tests are provided and can be run from the command `pytest` and its coverage extension.  Run the following in the terminal.

* `pytest --cov=forecast_tools tests/`

**All contributions are welcome and must include unit tests!**