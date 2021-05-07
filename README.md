# Simultaneous Time Series Forecasting on the World's COVID-19 Daily Vaccinations

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/drkostas/data_mining/master/LICENSE)

## Table of Contents

+ [About](#about)
    + [Code Locations](#codeloc)
    + [Poster & Extended Abstract](#docloc)
    + [Information About The Dataset](#datasetinfo)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Setting Up](#installing)
+ [Running the code](#run_locally)
    + [Configuration](#configuration)
    + [Running Jupyter](#jupyter)
+ [Built With](#built_with)
+ [License](#license)
+ [Acknowledgments](#acknowledgments)

## About <a name = "about"></a>

Dataset: [COVID-19 World Vaccination Progress](https://www.kaggle.com/gpreda/covid-world-vaccination-progress)
<br>
This is my project for the Data Mining Course (COSC-526). The main code is in
this [Jupyter Notebook](project.ipynb).

### Code Locations <a name = "codeloc"></a>

- The dataset is in the datasets/covid-world-vaccinations-progress directory
- The metadata dataset is in the datasets/countries-of-the-world directory
- The jupyter notebook used is the [project.ipynb](project.ipynb)
- Some custom packages used in the notebook are located in the [data_mining directory](data_mining):
    - Project Utils:
        - NullsFixer: for inferring the nulls in the COVID-19 vaccination dataset
        - Preprocess: the preprocessing code of the dataset before training
        - BuildModel: contains all the functions related to the building of the TF model
        - Visualizer: the implementations of all the visualizations
    - Configuration: it handles the yml configuration
    - ColorizedLogger: code for formatted logging that saves output in log files
    - timeit: ContextManager+Decorator for timing functions and code blocks
- The project was compiled using my Template Cookiecutter project: https://github.com/drkostas/starter

### Document Locations <a name = "docloc"></a>

The extended abstract and the poster are both located in the [Documents folder](Documents).

### Information About The Dataset <a name = "datasetinfo"></a>

The [COVID-19 Vaccination Progress Dataset](https://www.kaggle.com/gpreda/covid-world-vaccination-progress)
contains information about the daily and total vaccinations of 193 different countries over 135
different dates. The data are being collected almost daily and of writing this (4/29), the dataset has
14230 rows and 15 different features.

The features of the dataset are the following:

- **Country**: this is the country for which the vaccination information is provided
- **Country ISO Code**: ISO code for the country
- **Date**: date for the data entry; for some dates we have only the daily vaccinations, for others, only
  the (cumulative) total
- **Total number of vaccinations**: this is the absolute number of total immunizations in the country
- **Total number of people vaccinated**: a person, depending on the immunization scheme, will receive one
  or more (typically 2) vaccines; at a certain moment, the number of vaccination might be larger than
  the number of people
- **Total number of people fully vaccinated**: this is the number of people that received the entire set of
  immunization according to the immunization scheme (typically 2); at a certain moment in time, there
  might be a certain number of people that received one vaccine and another number (smaller) of people
  that received all vaccines in the scheme
- **Daily vaccinations (raw)**: for a certain data entry, the number of vaccination for that date/country
- **Daily vaccinations**: for a certain data entry, the number of vaccination for that date/country
- **Total vaccinations per hundred**: ratio (in percent) between vaccination number and total population up
  to the date in the country
- **Total number of people vaccinated per hundred**: ratio (in percent) between population immunized and
  total population up to the date in the country
- **Total number of people fully vaccinated per hundred**: ratio (in percent) between population fully
  immunized and total population up to the date in the country
- **Number of vaccinations per day**: number of daily vaccination for that day and country
- **Daily vaccinations per million**: ratio (in ppm) between vaccination number and total population for
  the current date in the country
- **Vaccines used in the country**: total number of vaccines used in the country (up to date)
- **Source name**: source of the information (national authority, international organization, local
  organization etc.)
- **Source website**: website of the source of information

For recalculating the _per hundred people values_ we used another dataset that contains some metadata
about the countries of the world, including their **population**.<br>
Metadata
Dataset: [DataBank - World Development Indicators](https://databank.worldbank.org/source/world-development-indicators#)

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your machine.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python >= 3.6 and any Bash based shell (e.g. zsh) installed.

```ShellSession

$ python3.6 -V
Python 3.6.13

$ echo $SHELL
/usr/bin/zsh

```

## Setting Up <a name = "installing"></a>

All the installation steps are being handled by the [Makefile](Makefile). The `server=local` flag
basically specifies that you want to use conda instead of venv, and it can be changed easily in the
lines `#25-28`. `local`  is also the default flag, so you can omit it.

```ShellSession
$ make install server=local
```

To update the COVID-19 vaccination dataset with the latest information, run:

```ShellSession
$ make download_dataset server=local
```

## Running the code <a name = "run_locally"></a>

In order to run the code, you will only need to modify the yml file if you need to, and open a jupyter
server.

### Modifying the Configuration <a name = "configuration"></a>

There is an already configured yml file under [confs/covid.yml](confs/covid.yml) with the following
structure:

```yaml
tag: project
covid-progress:
  - properties:
      data_path: datasets/covid-world-vaccination-progress/country_vaccinations.csv
      data_extra_path: datasets/world-bank/data.csv
      log_path: logs/covid_progress.log
    type: csv
```

### Running Jupyter <a name = "jupyter"></a>

After loading the cond environment with the command `conda activate data_mining`, run
`jupyter notebook` and open the [project.ipynb](project.ipynb) file.

## TODO <a name = "todo"></a>

Read the [TODO](TODO.md) to see the current task list.

## Built With <a name = "built_with"></a>

* [Jupyter](https://jupyter.org/) - An interactive computing framework
* [Tensorflow](https://jupyter.org/) - A deep learning framework

## License <a name = "license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments <a name = "acknowledgments"></a>

* Thanks to PurpleBooth for
  the [README template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)

