import configargparse
from .run_training import run_training as run_main

def main():
    parser = configargparse.ArgParser(
            description="Mask training module",
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            allow_abbrev=False
        )

    parser.add_argument("-c", "--config", type=str, required=True, is_config_file=True, help="path to YAML config file")
    parser.add_argument("-d", "--dir", type=str, help="path to training data")
    parser.add_argument("--epochs", type=int, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, help="learning rate"),
    parser.add_argument("--val_ratio", type=float, help="validation split ratio")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--crop_size", type=int, help="crop size (square)")
    parser.add_argument("--num_workers", type=int, help="number of workers")
    parser.add_argument("--log_image_interval", type=int, help="step interval to log images to TB")
    parser.add_argument("--log_interval", type=int, help="step interval to log data to console")
    parser.add_argument("--unet_depth", type=int, help="U-Net layers")
    parser.add_argument("--num_fmaps", type=int, help="Number of U-Net feature maps")
    args = parser.parse_args()


    run_main(
        args.dir, 
        args.epochs,
        args.learning_rate,
        args.val_ratio,
        args.batch_size,
        args.crop_size,
        args.num_workers,
        args.log_image_interval,
        args.log_interval,
        args.unet_depth,
        args.num_fmaps
        )

if __name__ == "__main__":
    main()