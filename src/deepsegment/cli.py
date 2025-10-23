import configargparse
from .run_training import run_training as run_main

def main():
    parser = configargparse.ArgParser(
            description="Mask training module",
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            allow_abbrev=False
        )

    parser.add_argument("-c", "--config", type=str, required=True, is_config_file=True, help="Path to YAML config file.")
    parser.add_argument("-d", "--dir", type=str, help="Path to training data.")
    args = parser.parse_args

    run_main(args)

if __name__ == "__main__":
    main()