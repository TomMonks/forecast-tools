# Contributing

Please fork Dev, make your modifications, run the unit tests and submit a pull request to dev for review.

Development environment:

```
mamba env create -f binder/environment.yml
```

```
mamba activate forecast_tools
```

Unit tests are provided and can be run via `hatch` and its coverage extension.  Run the following in the terminal.

To run tests in multiple Python environments (3.9-3.12)

```
hatch test --all
```

To obtain a coverage report run

```
hatch test --cover
```

**All contributions are welcome and must include unit tests!**