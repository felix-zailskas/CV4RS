import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.pytorch_models import ResNet50
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda



def train():
	# distr_type = "countries"
	distr_type = "countries_random"
	csv_paths = [str(p) for p in Path(f'data/{distr_type}/').glob('*train*.csv')]
	csv_paths = csv_paths[:3]
	cuda_no = 1
	batch_size = 128
	num_workers = 0
	epochs = 2
	communication_rounds = 1

	channels = 10
	num_classes = 19
	# model = create_mlp_mixer(channels, num_classes)
	# model = create_convmixer(channels=channels, num_classes=num_classes, pretrained=False)
	# model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)
	global_client = GlobalClient(
		model=model,
		lmdb_path="/faststorage/BigEarthNet_S1_S2/BEN_S1_S2.lmdb",
		val_path=f"data/{distr_type}/all_test.csv",
		csv_paths=csv_paths,
	)
	global_model, global_results = global_client.train(communication_rounds=communication_rounds, epochs=epochs)
	print(global_results)


if __name__ == '__main__':
    train()