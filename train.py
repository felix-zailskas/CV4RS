import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import time
import torch
from utils.pytorch_models import ResNet18
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer_1024_20
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda



def train():
	csv_paths = [str(p) for p in Path('data/countries/').glob('*train.csv')]
	cuda_no = 1
	batch_size = 128
	num_workers = 0
	epochs = 1
	communication_rounds = 1

	channels = 10
	num_classes = 19
	dataset_filter = "serbia"
	mlp_mixer = create_mlp_mixer(channels, num_classes)
	global_client_mlp_mixer = GlobalClient(
	    model=mlp_mixer,
	    lmdb_path="/faststorage/BigEarthNet_S1_S2/BEN_S1_S2.lmdb",
	    val_path="data/countries/all_test.csv",
	    csv_paths=csv_paths,
	)
	global_mlp_mixer_results, global_mlp_mixer_client_results = global_client_mlp_mixer.train(communication_rounds=communication_rounds, epochs=epochs)



if __name__ == '__main__':
    train()