from pathlib import Path
import configparser
import argparse
from pathlib import Path
from model import Model

param_file = "params.ini"

if __name__ == "__main__":
    params = {}
    config = configparser.ConfigParser()
    config.read(param_file)
    params["iterations"] = config["general"]["iterations"]
    params["initial_burn_in"] = config["general"]["initial_burn_in"]
    params["compute_perf"] = int(config["general"]["compute_perf"])
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dataset_path", action="store", type=str, required=True)
    parser.add_argument("--burn", action="store", type=int, default=0)
    parser.add_argument("--thinning", action="store", type=int, default=1)
    parser.add_argument("-d", "--dim", action="store", type=int, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--e_threshold", action="store", type=float, default=0)
    parser.add_argument("--cascade", action="store_true")
    args = parser.parse_args()
    # if hyper parameters file exists load them, use default values otherwise
    hyperparam_path = (
        Path(args.dataset_path) / "models" / str(args.dim) / "hyperparameters"
    )
    if hyperparam_path.exists():
        hyperparameters = configparser.ConfigParser()
        hyperparameters.read(hyperparam_path)
        hyperparameters = hyperparameters["params"]
        print("Hyper parameters loaded:", hyperparameters)
    else:
        hyperparameters = {
            "alpha1": 0.5,
            "alpha2": 0.5,
            "sigmaC": 1,
            "sigmaR": 1,
        }
    model = Model(args)
    # Train
    if args.train:
        model.train(params, hyperparameters)
    # Test
    elif args.test:
        e_threshold = float(args.e_threshold)
        burn_in, thinning = args.burn, args.thinning
        model.test(burn_in, thinning, e_threshold)
    # Print cascades according to ranks of nodes
    elif args.cascade:
        burn_in, thinning = args.burn, args.thinning
        model.print_cascades(burn_in, thinning)
    else:
        print("Specify --train or --test or --cascade")
        exit()
