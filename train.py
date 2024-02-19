import argparse
import datetime
import multiprocessing as mp
import os
from pathlib import Path

from logger.logger import CustomLogger
from models.ConvMixer import create_convmixer
from models.MLPMixer import _create_mixer
from models.poolformer import create_poolformer_s12
from utils.clients import GlobalClientFedAvg
from utils.clients_feddc import GlobalClientFedDC
from utils.pytorch_models import ResNet50

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


LOCAL_EPOCHS = 3  # amount of epochs each client trains for locally
GLOBAL_COMMUNICATION_ROUNDS = 20  # amount of communication rounds the global model
NUM_CHANNELS = 10
NUM_CLASSES = 19


def train(args):
    # processing input args
    input_args = []
    # dataset split
    if args.DS == 1:
        # data is split randomly => lowest non-IID
        distr_type = "scenario1"
    elif args.DS == 2:
        # data is split by country => medium non-IID
        distr_type = "scenario2"
    elif args.DS == 3:
        # data is split by country and season => highest non-IID
        distr_type = "scenario3"
    else:
        raise ValueError("Please specify Dataset")
    input_args.append(distr_type)
    # averaging algorithm
    if args.algo == "fedavg":
        avg_algorithm = "fedavg"
    else:
        raise ValueError("Please specify averaging algorithm")
    input_args.append(avg_algorithm)
    # used model type
    if args.model == "mlpmixer":
        model_args = dict(
            patch_size=8,
            num_blocks=8,
            embed_dim=512,
            img_size=120,
            num_classes=NUM_CLASSES,
            in_chans=NUM_CHANNELS,
        )  # best results
        model = _create_mixer("mixer_s16_224", pretrained=False, **model_args)
        model_name = args.model
    elif args.model == "convmixer":
        model = create_convmixer(
            channels=NUM_CHANNELS, num_classes=NUM_CLASSES, pretrained=args.pretrained
        )
        model_name = args.model
    elif args.model == "poolformer":
        model = create_poolformer_s12(
            layers=[2, 2, 6, 2], in_chans=NUM_CHANNELS, num_classes=NUM_CLASSES
        )
        model_name = args.model
    elif args.model == "resnet":
        model = ResNet50(
            "ResNet50", channels=NUM_CHANNELS, num_cls=NUM_CLASSES, pretrained=False
        )
        model_name = "resnet50"
    else:
        raise ValueError("Passed model name is not defined")
    input_args.append(model_name)
    # used algorithm
    if args.algo == "fedavg":
        algorithm = "fedavg"
    elif args.algo == "moon":
        algorithm = "moon"
    elif args.algo == "feddc":
        algorithm = "feddc"
    input_args.append(algorithm)

    selected_args_str = "/".join(input_args)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        selected_args_str
        + f"/epochs({LOCAL_EPOCHS})_comrounds({GLOBAL_COMMUNICATION_ROUNDS})_{current_time}"
    )
    if args.name is not None:
        run_name += f"_{args.name}"
    global_logger = CustomLogger("GlobalLogger", f"./logs/{run_name}")
    global_logger.info(f"Using model: {type(model)}")
    global_logger.info(f"Using Dataset: {distr_type}")
    global_logger.info(f"Using pretrained weights: {args.pretrained}")

    # check for feddc use
    if algorithm == "feddc" and not args.feddc:
        global_logger.error(
            "Cannot use faulty implementation of FedDC without the flag `--feddc`"
        )
        exit()
    elif algorithm == "moon":
        global_logger.error("MOON currently not implemented")
        exit()

    # setting training parameters
    csv_paths = [str(p) for p in Path(f"data/{distr_type}/").glob("*train*.csv")]
    if algorithm == "fedavg":
        global_client = GlobalClientFedAvg(
            model=model,
            lmdb_path="/faststorage/BigEarthNet_S1_S2/BEN_S1_S2.lmdb",
            val_path=f"data/{distr_type}/all_test.csv",
            csv_paths=csv_paths,
            name=f"GlobalModel_{args.model}",
            logger=global_logger,
            run_name=run_name,
        )
    elif algorithm == "feddc":
        global_client = GlobalClientFedDC(
            model=model,
            lmdb_path="/faststorage/BigEarthNet_S1_S2/BEN_S1_S2.lmdb",
            val_path=f"data/{distr_type}/all_test.csv",
            csv_paths=csv_paths,
            name=f"GlobalModel_{args.model}",
            logger=global_logger,
            run_name=run_name,
        )
    global_model, global_results = global_client.train(
        communication_rounds=GLOBAL_COMMUNICATION_ROUNDS, epochs=LOCAL_EPOCHS
    )
    global_logger.info(global_results)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("-DS", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument(
        "--algo", type=str, default="fedavg", choices=["fedavg", "feddc", "moon"]
    )
    parser.add_argument("--feddc", action="store_true")
    parser.add_argument("--pretrained", action="store_true")  # default is False
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mlpmixer", "convmixer", "poolformer", "resnet"],
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    train(args)
