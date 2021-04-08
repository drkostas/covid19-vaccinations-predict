# DataMiningProject<img src='https://github.com/CISC879-BigData/Project_Georgiou/blob/master/img/snek.png' align='right' width='180' height='104'>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/drkostas/data_mining/master/LICENSE)

## Table of Contents

+ [About](#about)
    + [Information About The Dataset](#datasetinfo)
    + [Questions To Be Answered](#questionsinfo)
    + [Some Details](#detailsinfo)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Installing, Testing, Building](#installing)
    + [Available Make Commands](#check_make_commamnds)
    + [Clean Previous Builds](#clean_previous)
    + [Create a new virtual environment](#create_env)
    + [Build Locally (and install requirements)](#build_locally)
    + [Run the tests](#tests)
+ [Running locally](#run_locally)
    + [Configuration](#configuration)
    + [Environment Variables](#env_variables)
    + [Execution Options](#execution_options)
        + [DataMiningProject Main](#data_mining_main)
        + [DataMiningProject Greet CLI](#data_mining_cli)
+ [Deployment](#deployment)
+ [Continuous Ιntegration](#ci)
+ [Todo](#todo)
+ [Built With](#built_with)
+ [License](#license)
+ [Acknowledgments](#acknowledgments)

## About <a name = "about"></a>

Dataset: [COVID-19 World Vaccination Progress](https://www.kaggle.com/gpreda/covid-world-vaccination-progress)
<br>
This is my project for the Data Mining Course (COSC-526). The main code is in this [Jupyter Notebook](project.ipynb).

### Information About The Dataset <a name = "datasetinfo"></a>
This dataset contains information about the vaccinations happening in each country daily. The data are being collected almost daily from this website using this code. As of writing this (2/27), the dataset has 4,380 rows with vaccination data for 112 unique countries and is in the CSV format.

It has 15 columns in total, including among others the country name, the daily vaccination, the vaccinated people per million that date, and the source of each record.

### Questions To Be Answered <a name = "questionsinfo"></a>
- Can you identify countries that faced bottlenecks on their daily vaccination rates?
- Can you cluster together countries that faced similar bottlenecks? In what sense are they related?
- Can you enrich the data with more info (country location, GDP, etc) to achieve better results on the previous question?
- Can you track down the bottlenecks and find patterns in how they propagate from day to day from one cluster to another?
- Can you predict future bottlenecks on some clusters based on these patterns?

### Some Details <a name = "detailsinfo"></a>

- The dataset is in the datasets/covid-world-vaccinations-progress directory
- In the data mining directory are located three custom packages:
  - Configuration: for handling the yml configuration
  - ColorizedLogger: For formatted logging that saves output in log files
  - timeit: ContextManager&Decorator for timing functions and code blocks
- The project was compiled using my Template Cookiecutter project: https://github.com/drkostas/starter

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes. See deployment for notes on how to deploy the project on a live
system.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python > 3.6 and any Bash based shell (e.g. zsh) installed.

```ShellSession

$ python3.8 -V
Python 3.8.5

$ echo $SHELL
/usr/bin/zsh

```

## Installing, Testing, Building <a name = "installing"></a>

All the installation steps are being handled by the [Makefile](Makefile). The `server=local` flag
basically specifies that you want to use conda instead of venv, and it can be changed easily in the
lines `#25-28`. `local`  is also the default flag, so you can omit it.

<i>If you don't want to go through the detailed setup steps but finish the installation and run the
tests quickly, execute the following command:</i>

```ShellSession
$ make install server=local
```

To update the Covid Dataset, run:
```ShellSession
$ make download_dataset server=local
```


<i>If you executed the previous command, you can skip through to
the [Running locally section](#run_locally).</i>

### Check the available make commands <a name = "check_make_commamnds"></a>

```ShellSession

$ make help
-----------------------------------------------------------------------------------------------------------
                                              DISPLAYING HELP                                              
-----------------------------------------------------------------------------------------------------------
Use make <make recipe> [server=<prod|circleci|local>] to specify the server
Prod, and local are using conda env, circleci uses virtualenv. Default: local

make help
       Display this message
make install [server=<prod|circleci|local>]
       Call clean delete_conda_env create_conda_env setup run_tests
make clean [server=<prod|circleci|local>]
       Delete all './build ./dist ./*.pyc ./*.tgz ./*.egg-info' files
make delete_env [server=<prod|circleci|local>]
       Delete the current conda env or virtualenv
make create_env [server=<prod|circleci|local>]
       Create a new conda env or virtualenv for the specified python version
make setup [server=<prod|circleci|local>]
       Call setup.py install
make run_tests [server=<prod|circleci|local>]
       Run all the tests from the specified folder
-----------------------------------------------------------------------------------------------------------

```

### Clean any previous builds <a name = "clean_previous"></a>

```ShellSession
$ make clean delete_env server=local
```

### Create a new virtual environment <a name = "create_env"></a>

For creating a conda virtual environment run:

```ShellSession
$ make create_env server=local 
```

### Build Locally (and install requirements) <a name = "build_locally"></a>

To build the project locally using the setup.py install command (which also installs the requirements),
execute the following command:

```ShellSession
$ make setup server=local
```

### Run the tests <a name = "tests"></a>

The tests are located in the `tests` folder. To run all of them, execute the following command:

```ShellSession
$ make run_tests server=local
```

## Running the code locally <a name = "run_locally"></a>

In order to run the code, you will only need to change the yml file if you need to, and either run its
file directly or invoke its console script.

<i>If you don't need to change yml file, skip to [Execution Options](#execution_options).

### Modifying the Configuration <a name = "configuration"></a>

There is an already configured yml file under [confs/template_conf.yml](confs/template_conf.yml) with
the following structure:

```yaml
tag: template
example_db:
  - config:
      hostname: example.host.name
      username: my_name
      password: !ENV ${PASS}
      db_name: my_db1
      port: 3306
    type: mysql
```

The `!ENV` flag indicates that you are passing an environmental value to this attribute. You can change
the values/environmental var names as you wish. If a yaml variable name is changed/added/deleted, the
corresponding changes should be reflected on the [yml_schema.json](configuration/yml_schema.json) too
which validates it.

### Set the required environment variables <a name = "env_variables"></a>

In order to run the [main.py](data_mining/main.py)  you will need to set the
environmental variables you are using in your configuration yml file. Example:

```ShellSession
$ export PASS=my_password
```

The best way to do that, is to create a .env file ([example](env_example)), and source it before
running the code.

### Execution Options <a name = "execution_options"></a>

First, make sure you are in the correct virtual environment:

```ShellSession
$ conda activate data_mining

$ which python
/home/drkostas/anaconda3/envs/data_mining/bin/python

```

#### DataMiningProject Main <a name = "data_mining"></a>

Now, in order to run the code you can either call the [main.py](data_mining/main.py)
directly, or invoke the `data_mining_main`
console script.

```ShellSession
$ python data_mining/main.py --help
usage: main.py -c CONFIG_FILE [-m {run_mode_1,run_mode_2,run_mode_3}] [-l LOG] [-d] [-h]

This is my project for the Data Mining Course (COSC-526)

Required Arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The configuration yml file

Optional Arguments:
  -m {run_mode_1,run_mode_2,run_mode_3}, --run-mode {run_mode_1,run_mode_2,run_mode_3}
                        Description of the run modes
  -l LOG, --log LOG     Name of the output log file
  -d, --debug           Enables the debug log messages
  -h, --help            Show this help message and exit


# Or

$ data_mining_main --help
usage: main.py -c CONFIG_FILE [-m {run_mode_1,run_mode_2,run_mode_3}] [-l LOG] [-d] [-h]

This is my project for the Data Mining Course (COSC-526)

Required Arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The configuration yml file

Optional Arguments:
  -m {run_mode_1,run_mode_2,run_mode_3}, --run-mode {run_mode_1,run_mode_2,run_mode_3}
                        Description of the run modes
  -l LOG, --log LOG     Name of the output log file
  -d, --debug           Enables the debug log messages
  -h, --help            Show this help message and exit
```

#### DataMiningProject CLI <a name = "data_mining_cli"></a>

There is also a [cli.py](data_mining/cli.py) which you can also invoke it by its
console script too
(`cli`).

```ShellSession
$ cli --help
Usage: cli [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.

Commands:
  bye
  hello
```

## Deployment <a name = "deployment"></a>

The deployment is being done to <b>Heroku</b>. For more information you can check
the [setup guide](https://devcenter.heroku.com/articles/getting-started-with-python).

Make sure you check the
defined [Procfile](Procfile) ([reference](https://devcenter.heroku.com/articles/getting-started-with-python#define-a-procfile))
and that you set
the [above-mentioned environmental variables](#env_variables) ([reference](https://devcenter.heroku.com/articles/config-vars))
.

## Continuous Integration <a name = "ci"></a>

For the continuous integration, the <b>CircleCI</b> service is being used. For more information you can
check the [setup guide](https://circleci.com/docs/2.0/language-python/).

Again, you should set
the [above-mentioned environmental variables](#env_variables) ([reference](https://circleci.com/docs/2.0/env-vars/#setting-an-environment-variable-in-a-context))
and for any modifications, edit the [circleci config](/.circleci/config.yml).

## TODO <a name = "todo"></a>

Read the [TODO](TODO.md) to see the current task list.

## Built With <a name = "built_with"></a>

* [Heroku](https://www.heroku.com) - The deployment environment
* [CircleCI](https://www.circleci.com/) - Continuous Integration service

## License <a name = "license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments <a name = "acknowledgments"></a>

* Thanks to PurpleBooth for
  the [README template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)

