import argparse
import datetime
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from logger.logger import CustomLogger
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from models.poolformer import create_poolformer_s12
from utils.clients import GlobalClient
from utils.pytorch_models import ResNet50
from utils.pytorch_utils import start_cuda

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


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
    # used model type
    if args.model == "mlpmixer":
        model = create_mlp_mixer(channels, num_classes)
    elif args.model == "convmixer":
        model = create_convmixer(
            channels=channels, num_classes=num_classes, pretrained=False
        )
    elif args.model == "poolformer":
        model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
    elif args.model == "resnet":
        model = ResNet50(
            "ResNet50", channels=channels, num_cls=num_classes, pretrained=False
        )
    else:
        raise ValueError("Passed model name is not defined")
    input_args.append(args.model)
    
    selected_args_str = "_".join(input_args)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = selected_args_str + "_" + current_time
    global_logger = CustomLogger("GlobalLogger", f"./logs/{run_name}")
    global_logger.info(f"Using model: {type(model)}")
    global_logger.info(f"Using Dataset: {distr_type}")

    # setting training parameters
    csv_paths = [str(p) for p in Path(f"data/{distr_type}/").glob("*train*.csv")]
    cuda_no = 1
    batch_size = 128
    num_workers = 0
    epochs = 1
    communication_rounds = 1

    channels = 10
    num_classes = 19

    global_client = GlobalClient(
        model=model,
        lmdb_path="/faststorage/BigEarthNet_S1_S2/BEN_S1_S2.lmdb",
        val_path=f"data/{distr_type}/all_test.csv",
        csv_paths=csv_paths,
        name=f"GlobalModel_{args.model}",
        logger=global_logger,
        run_name=run_name,
    )
    global_model, global_results = global_client.train(
        communication_rounds=communication_rounds, epochs=epochs
    )
    global_logger.info(global_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-DS", type=int, default=None, choices=[1, 2])
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mlpmixer", "convmixer", "poolformer", "resnet"],
    )
    args = parser.parse_args()

    train(args)
