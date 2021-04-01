import traceback
import argparse

from data_mining import Configuration, ColorizedLogger, timeit

basic_logger = ColorizedLogger(logger_name='Main', color='yellow')
fancy_logger = ColorizedLogger(logger_name='FancyMain',
                               color='blue',
                               on_color='on_red',
                               attrs=['underline', 'reverse', 'bold'])


def get_args() -> argparse.Namespace:
    """Setup the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='A template for python projects.',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The configuration yml file"
    }
    required_args.add_argument('-c', '--config-file', **config_file_params)
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument('-m', '--run-mode', choices=['run_mode_1', 'run_mode_2', 'run_mode_3'],
                               default='run_mode_1',
                               help='Description of the run modes')
    optional_args.add_argument('-l', '--log',
                               help="Name of the output log file")
    optional_args.add_argument('-d', '--debug', action='store_true',
                               help='Enables the debug log messages')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


@timeit(custom_print="{func_name} took {duration:2.5f} to run!")
def main():
    """This is the main function of main.py

    Example:
        python data_mining/main.py -m run_mode_1
                                                     -c confs/template_conf.yml
                                                     -l logs/output.log
    """

    # Initializing
    args = get_args()
    ColorizedLogger.setup_logger(log_path=args.log, debug=args.debug)
    # Load the configuration
    # configuration = Configuration(config_src=args.config_file,
    #                               config_schema_path='yml_schema_strict.json')
    configuration = Configuration(config_src=args.config_file)
    # Prints
    basic_logger.info("Starting in run mode: {0}".format(args.run_mode))
    basic_logger.info("Examples:")
    fancy_logger.info("You can customize the logger like this")
    fancy_logger.info("You can customize each log message differently",
                      color="green", on_color="on_white", attrs=[])
    basic_logger.info("If you want to print complete blank lines use nl(num_lines=<#>):")
    basic_logger.nl(num_lines=2)
    # Example timeit code block
    basic_logger.info("Lastly, you can use timeit either as a function Wrapper or a ContextManager:")
    custom_print = "Iterating in a 10,000-number-range took {duration:2.5f} seconds."
    with timeit(custom_print=custom_print):
        for _ in range(10000):
            pass


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        basic_logger.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
