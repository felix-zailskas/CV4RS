import pandas as pd
from sklearn.model_selection import train_test_split
import timm
import torch

cuda_no = 1
batch_size = 128
num_workers = 0
epochs = 20
communication_rounds = 10

channels = 10
num_classes = 19
dataset_filter = "serbia"

from utils.pytorch_models import ResNet18
from models.poolformer import create_poolformer_s12
from utils.clients import GlobalClient

resnet18 = ResNet18(num_cls=num_classes, channels=channels, pretrained=True)
convmixer = timm.create_model('convmixer_1024_20_ks9_p14', pretrained=True, in_chans=channels, num_classes=num_classes)
device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
convmixer.to(device)

global_client_resnet18 = GlobalClient(
    model=resnet18,
    lmdb_path="data/BigEarth_Serbia_Summer_S2.lmdb",
    val_path="data/test.csv",
    csv_paths=["data/c1_train.csv", "data/c2_train.csv", "data/c3_train.csv"],
)
global_client_convmixer = GlobalClient(
    model=convmixer,
    lmdb_path="data/BigEarth_Serbia_Summer_S2.lmdb",
    val_path="data/test.csv",
    csv_paths=["data/c1_train.csv", "data/c2_train.csv", "data/c3_train.csv"],
)

global_resnet18_results = global_client_convmixer.train(communication_rounds=communication_rounds, epochs=epochs)