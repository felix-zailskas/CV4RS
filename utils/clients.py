import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        csv_path: list[str],
        batch_size: int = 128,
        num_workers: int = 0,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        device: torch.device = torch.device('cpu')
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.dataset = Ben19Dataset(lmdb_path, csv_path)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.device = device

    def set_model(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)

    def train_one_round(self, epochs: int):
        state_before = copy.deepcopy(self.model.state_dict())

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)

            self.train_epoch()

        state_after = self.model.state_dict()

        model_update = {}
        for key, value_before in state_before.items():
            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(
                torch.DoubleTensor
            )
            model_update[key] = diff

        return model_update

    def train_epoch(self):
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data, labels, index = batch["data"], batch["label"], batch["index"]
            data = data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()


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
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)
        self.clients = [
            FLCLient(copy.deepcopy(self.model), lmdb_path, csv_path, device=self.device)
            for csv_path in csv_paths
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

    def train(self, communication_rounds: int, epochs: int):
        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            self.communication_round(epochs)
            report = self.validation_round()

            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            for client in self.clients:
                client.set_model(self.model)
        
        return self.results

    def validation_round(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch["data"].to(self.device)
                labels = batch["label"].numpy()

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)

                y_true += list(labels)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)

        y_true = np.asarray(y_true)
        report = get_classification_report(
            y_true, y_predicted, predicted_probs, self.dataset_filter
        )
        return report

    def communication_round(self, epochs: int):
        # here the clients train
        # TODO: could be parallelized
        model_updates = [client.train_one_round(epochs) for client in self.clients]

        # parameter aggregation
        update_aggregation = self.aggregator.fed_avg(model_updates)

        # update the global model
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)
