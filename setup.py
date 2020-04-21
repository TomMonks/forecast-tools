import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forecast-tools-TomMonks", 
    version="0.0.1",
    author="Thomas Monks",
    description="Tools to support the forecasting process in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomMonks/forecast-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)