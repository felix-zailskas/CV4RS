import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import threading
from datetime import datetime
from pathlib import Path 

# from timm.models.convmixer import ConvMixer
from models.ConvMixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50
from logger.logger import CustomLogger
from utils.pytorch_datasets import Ben19Dataset
from utils.pytorch_utils import (
    get_classification_report,
    init_results,
    print_micro_macro,
    update_results,
    start_cuda
)



class Aggregator:
    def __init__(self) -> None:
        pass

    def fed_avg(self, model_updates: list[dict]):
        assert len(model_updates) > 0, "Trying to aggregate empty update list"
        
        update_aggregation = {}
        for key in model_updates[0].keys():
            params = torch.stack([update[key] for update in model_updates], dim=0)
            avg = torch.mean(params, dim=0)
            update_aggregation[key] = avg
        
        return update_aggregation


class FLCLient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_path: list[str],
        batch_size: int = 128,
        num_workers: int = 0,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device('cpu'),
        dataset_filter: str = "serbia",
        logger: CustomLogger = None,
        name: str = "FLClient",
        run_name: str = ""
    ) -> None:
        self.name = name
        self.logger = logger if logger is not None else CustomLogger(self.name, log_dir=f"./logs/{run_name}")
        self.model = model
        self.previous_model = model
        self.global_model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)
        self.dataset = Ben19Dataset(lmdb_path, csv_path)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.device = device

        self.validation_set = Ben19Dataset(
            lmdb_path=lmdb_path, csv_path=val_path, img_transform="default"
        )
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def set_model(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)

    def train_one_round(self, epochs: int, training_device = None, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())
        self.global_model = copy.deepcopy(self.model) # save current model as global model
        if training_device is None:
            self.model.to(self.device)
            self.global_model.to(self.device)
        else:
            self.logger.info(f"Putting model {self.name} onto {training_device}")
            self.model.to(training_device)
            self.global_model.to(training_device)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            self.logger.info("Epoch {}/{}".format(epoch, epochs))

            self.train_epoch(training_device)
        self.logger.info("Training done!")
        if validate:
            self.logger.info("validation started")
            report = self.validation_round(training_device)
            self.results = update_results(self.results, report, self.num_classes)
            self.logger.info("Validation done!")
        
        state_after = self.model.state_dict()
        self.previous_model = copy.deepcopy(self.model) #save previous model for next iteration

        model_update = {}
        for key, value_before in state_before.items():
            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(
                torch.DoubleTensor
            )
            model_update[key] = diff

        return model_update

    def change_sizes(self, labels):
        new_labels=np.zeros((len(labels[0]),19))
        for i in range(len(labels[0])): #128
            for j in range(len(labels)): #19
                new_labels[i,j] =  int(labels[j][i])
        return new_labels
    
    def train_epoch(self, training_device = None):
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc=f"training {self.name} on device {training_device if training_device is not None else self.device}")):
            self.logger.info(f"Batch {idx + 1}/{len(self.train_loader)}")
            data, labels, index = batch["data"], batch["label"], batch["index"]
            if training_device is None:
                data = data.to(self.device)
            else:
                data = data.to(training_device)
            label_new=np.copy(labels)
            label_new=self.change_sizes(label_new)
            if training_device is None:
                label_new = torch.from_numpy(label_new).to(self.device)
            else:
                label_new = torch.from_numpy(label_new).to(training_device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()
            # ------ MOON INITIAL IMPLEMENTATION ----------
            """
            data, labels, index = batch["data"], batch["label"], batch["index"]
            # data = data.cuda()
            if training_device is None:
                data = data.to(self.device)
            else:
                data = data.to(training_device)
            label_new=np.copy(labels)
            label_new=self.change_sizes(label_new)
            # label_new = torch.from_numpy(label_new).cuda()
            if training_device is None:
                label_new = torch.from_numpy(label_new).to(self.device)
            else:
                label_new = torch.from_numpy(label_new).to(training_device)
            self.optimizer.zero_grad()

            l_sup = torch.nn.CrossEntropyLoss()(self.model(data), label_new)
            z = self.model(data, features_only=True)
            z_global = self.global_model(data, features_only=True)
            z_prev = self.previous_model(data, features_only=True)

            exp1 = torch.exp(torch.nn.functional.cosine_similarity(z,z_global) / 0.5)  #temperature default:0.5 ? 
            exp2 = torch.exp(torch.nn.functional.cosine_similarity(z,z_prev) / 0.5)    #temperature default:0.5 ? 

            l_con = -torch.log(exp1 / (exp1 + exp2))

            loss = l_sup + 1 * l_con.sum() #µ default:1 ?

            #logits = self.model(data)
            #loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()
            """
    
    def validation_round(self, training_device = None):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                # data = batch["data"].to(self.device)
                if training_device is None:
                    data = batch["data"].to(self.device)
                else:
                    data = batch["data"].to(training_device)
                labels = batch["label"]
                label_new=np.copy(labels)
                label_new=self.change_sizes(label_new)

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)

                y_true += list(label_new)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)

        y_true = np.asarray(y_true)
        report = get_classification_report(
            y_true, y_predicted, predicted_probs, self.dataset_filter
        )
        return report
    
    def get_validation_results(self):
        return self.results

class GlobalClient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_paths: list[str],
        batch_size: int = 128,
        num_workers: int = 0,
        num_classes: int = 19,
        dataset_filter: str = "serbia",
        state_dict_path: str = None,
        results_path: str = None,
        logger: CustomLogger = None,
        name: str = "GlobalClient",
        run_name: str = ""
    ) -> None:
        self.name = name
        self.logger = logger if logger is not None else CustomLogger(self.name, f"./logs/{run_name}/")
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        self.logger.info(f'Using device: {self.device}')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)
        self.clients = [
            FLCLient(copy.deepcopy(self.model), lmdb_path, val_path, csv_path, num_classes=num_classes, dataset_filter=dataset_filter, device=self.device, name=f"FLClient_{i}", run_name=run_name)
            for i, csv_path in enumerate(csv_paths)
        ]
        self.validation_set = Ben19Dataset(
            lmdb_path=lmdb_path, csv_path=val_path, img_transform="default"
        )
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
        self.gpu_locks = [threading.Lock() for _ in range(torch.cuda.device_count())]
        
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if state_dict_path is None:
            if isinstance(model, ConvMixer):
                self.state_dict_path = f'checkpoints/global_convmixer_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.state_dict_path = f'checkpoints/global_mlpmixer_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.state_dict_path = f'checkpoints/global_poolformer_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.state_dict_path = f'checkpoints/global_resnet18_{dt}.pkl'

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f'results/convmixer_results_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.results_path = f'results/mlpmixer_results_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.results_path = f'results/poolformer_results_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.results_path = f'results/resnet18_results_{dt}.pkl'

    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()
        for com_round in range(1, communication_rounds + 1):
            self.logger.info("Round {}/{}".format(com_round, communication_rounds))

            self.communication_round(epochs)
            report = self.validation_round()

            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            for client in self.clients:
                client.set_model(self.model)
        self.train_time = time.perf_counter() - start

        self.client_results = [client.get_validation_results() for client in self.clients]
        self.save_results()
        self.save_state_dict()
        return self.results, self.client_results

    def change_sizes(self, labels):
        new_labels=np.zeros((len(labels[0]),19))
        for i in range(len(labels[0])): #128
            for j in range(len(labels)): #19
                new_labels[i,j] =  int(labels[j][i])
        return new_labels
    

    def validation_round(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch["data"].to(self.device)
                labels = batch["label"]
                label_new=np.copy(labels)
                label_new=self.change_sizes(label_new)

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)

                y_true += list(label_new)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)

        y_true = np.asarray(y_true)
        report = get_classification_report(
            y_true, y_predicted, predicted_probs, self.dataset_filter
        )
        return report

    def train_client_on_device(self, client, epochs, gpu_id):
        with torch.cuda.device(gpu_id):
            model_update = client.train_one_round(epochs, torch.device(gpu_id))
        return model_update
    
    def worker(self, client, gpu_id, model_updates, epochs):
        # with self.gpu_locks[gpu_id]:
        model_update = self.train_client_on_device(client, epochs, gpu_id)
        self.gpu_locks[gpu_id].release()
        model_updates.append(model_update)
        
    
    def communication_round(self, epochs: int):
        # here the clients train
        self.logger.info("Starting communication round")
        model_updates = []
        threads = []
        for client in self.clients:
            client_training = False
            while not client_training:
                for gpu_id in range(len(self.gpu_locks)):
                    if not self.gpu_locks[gpu_id].locked():
                        self.logger.info(f"{client.name} training on device #{gpu_id}")
                        self.gpu_locks[gpu_id].acquire()
                        thread = threading.Thread(target=self.worker, args=(client, gpu_id, model_updates, epochs))
                        threads.append(thread)
                        thread.start()
                        client_training = True
                        break
                
        for t in threads:
            t.join()
        self.logger.info("All clients done with training")
                
        # parameter aggregation
        update_aggregation = self.aggregator.fed_avg(model_updates)
        self.logger.info("Model update aggregation complete")
        # update the global model
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)
        self.logger.info("Communication round done")

    def save_state_dict(self):
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)  
        res = {'global':self.results, 'clients':self.client_results, 'train_time': self.train_time}
        torch.save(res, self.results_path)
