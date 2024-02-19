import copy
import itertools
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger.logger import CustomLogger
from utils.gpu_parallelization import GPUWorker, parallel_gpu_work
from utils.pytorch_datasets import Ben19Dataset
from utils.pytorch_utils import (
    change_sizes,
    get_classification_report,
    init_results,
    print_micro_macro,
    update_results,
)


def fed_avg(model_updates: list[dict]) -> dict:
    """
    Federated Averaging function. Takes a list of model parameters and computes the
    average of all of them with equal weight. Expects model updates to be from the
    same architecture.

    Args:
        model_updates (list[dict]): Model updates that should be averaged.

    Returns:
        dict: Averaged model updates.
    """
    assert len(model_updates) > 0, "Trying to aggregate empty update list"

    update_aggregation = {}
    for key in model_updates[0].keys():
        params = torch.stack([update[key] for update in model_updates], dim=0)
        avg = torch.mean(params, dim=0)
        update_aggregation[key] = avg

    return update_aggregation


class FLCLient:
    """
    Implementation of the local clients in the federated learning setup. The client can
    be initialized with any model type that extends the torch.nn.Module class.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_path: str,
        batch_size: int = 128,
        num_workers: int = 0,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "serbia",
        name: str = "FLClient",
        logger: CustomLogger = None,
        run_name: str = "",
    ) -> None:
        # logging metadata
        self.name = name
        self.logger = (
            logger
            if logger is not None
            else CustomLogger(self.name, f"./logs/{run_name}")
        )
        # training metadata
        self.model = model
        self.device = device
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        # training data
        self.results = init_results(self.num_classes)
        self.train_loader = DataLoader(
            Ben19Dataset(lmdb_path, csv_path),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            Ben19Dataset(
                lmdb_path=lmdb_path, csv_path=val_path, img_transform="default"
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def set_model(self, model: torch.nn.Module):
        """
        Set current model of this client.

        Args:
            model (torch.nn.Module): Model to set.
        """
        self.model = copy.deepcopy(model)

    def train_one_round(
        self, epochs: int, training_device: torch.device = None, validate: bool = False
    ) -> dict:
        """
        Performs one training iteration for this client including a specified amount of
        epochs.

        Args:
            epochs (int): Amount of epochs to train for.
            training_device (torch.device, optional): Device that the model should be
                trained on. If this is None than the device specified in the constructor
                will be used. Defaults to None.
            validate (bool, optional): If True the client will perform validation rounds
                of the local model. Defaults to False.

        Returns:
            dict: The model updates the local training iteration produced.
        """
        # save state before training to compute model updates later
        state_before = copy.deepcopy(self.model.state_dict())
        # put model on correct device
        if training_device is None:
            self.logger.info(f"Putting model {self.name} onto {self.device}")
            self.model.to(self.device)
        else:
            self.logger.info(f"Putting model {self.name} onto {training_device}")
            self.model.to(training_device)

        # initialize training methods
        self.optimizer = self.optimizer_constructor(
            self.model.parameters(), **self.optimizer_kwargs
        )
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)
        # training loop
        for epoch in range(1, epochs + 1):
            self.logger.info("Epoch {}/{}".format(epoch, epochs))

            self.train_epoch(training_device)
        self.logger.info("Training done!")

        # validation loop
        if validate:
            self.logger.info("validation started")
            report = self.validation_round(training_device)
            self.results = update_results(self.results, report, self.num_classes)
            self.logger.info("Validation done!")

        # compute model updates after training
        state_after = self.model.state_dict()
        model_update = {}
        for key, value_before in state_before.items():
            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(
                torch.DoubleTensor
            )
            model_update[key] = diff

        return model_update

    def train_epoch(self, training_device: torch.device = None):
        """
        Perform one epoch of training and update the parameters of this client's model.

        Args:
            training_device (torch.device, optional): Device that the model should be
                trained on. If this is None than the device specified in the constructor
                will be used. Defaults to None.
        """
        self.model.train()
        for _, batch in enumerate(
            tqdm(
                self.train_loader,
                desc=f"training {self.name} on device {training_device if training_device is not None else self.device}",
            )
        ):
            data, labels, _ = batch["data"], batch["label"], batch["index"]
            if training_device is None:
                data = data.to(self.device)
            else:
                data = data.to(training_device)
            label_new = np.copy(labels)
            label_new = change_sizes(label_new)
            if training_device is None:
                label_new = torch.from_numpy(label_new).to(self.device)
            else:
                label_new = torch.from_numpy(label_new).to(training_device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()

    def validation_round(self, training_device: torch.device = None) -> str | dict:
        """
        Perform validation round on client's model.

        Args:
            training_device (torch.device, optional): Device that the model should be
                trained on. If this is None than the device specified in the constructor
                will be used. Defaults to None.

        Returns:
            (str | dict): classification report of the current model state.
        """
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.val_loader, desc="test")):
                if training_device is None:
                    data = batch["data"].to(self.device)
                else:
                    data = batch["data"].to(training_device)
                labels = batch["label"]
                label_new = np.copy(labels)
                label_new = change_sizes(label_new)

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

    def get_validation_results(self) -> dict:
        """
        Return validation results created during training if validation flag was set.

        Returns:
            dict: Validation results of the clients training
        """
        return self.results


class GlobalClientFedAvg:
    """
    Implementation of the global client in the federated learning setup. The client can
    be initialized with any model type that extends the torch.nn.Module class.
    """

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
        name: str = "GlobalClient",
        logger: CustomLogger = None,
        run_name: str = "",
    ) -> None:
        # logging metadata
        self.name = name
        if run_name == "":
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger = (
            logger
            if logger is not None
            else CustomLogger(self.name, f"./logs/{run_name}")
        )

        # check for available GPUs
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            # validation is done on the first device
            self.device = torch.device(0)
            # GPU parallelization is active for more than one GPU
            self.gpu_parallelization = torch.cuda.device_count() > 1
        self.logger.info(f"Using device: {self.device}")

        # training metadata
        self.model = model
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)
        self.train_time = None
        # if gpu parallelization is active we only need the data loaders and no client objects
        if self.gpu_parallelization:
            self.train_loaders = [
                DataLoader(
                    Ben19Dataset(lmdb_path, csv_path),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    pin_memory=True,
                )
                for csv_path in csv_paths
            ]
        else:
            self.clients = [
                FLCLient(
                    copy.deepcopy(self.model),
                    lmdb_path,
                    val_path,
                    csv_path,
                    num_classes=num_classes,
                    dataset_filter=dataset_filter,
                    device=self.device,
                    name=f"FLClient_{i}",
                    run_name=run_name,
                )
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

        # set output paths if not specified by user
        if state_dict_path is None:
            self.state_dict_path = f"checkpoints/{run_name}.pkl"
        else:
            self.state_dict_path = state_dict_path
        if results_path is None:
            self.results_path = f"results/{run_name}.pkl"
        else:
            self.results_path = results_path

    def train(self, communication_rounds: int, epochs: int) -> tuple[dict, list[dict]]:
        """
        Perform the full training of the global model for the specified amount of
        communication rounds and epochs.

        Args:
            communication_rounds (int): Communication rounds to train for globally.
            epochs (int): Epochs to train for locally.

        Returns:
            tuple[dict, list[dict]]: Training results of this global model and all local clients.
        """
        self.comm_times = []
        train_start = time.perf_counter()
        for com_round in range(1, communication_rounds + 1):
            self.logger.info("Round {}/{}".format(com_round, communication_rounds))

            # communication round
            comm_start = time.perf_counter()
            if self.gpu_parallelization:
                self.parallel_communication_round(epochs)
            else:
                self.sequential_communication_round(epochs)
            comm_time = time.perf_counter() - comm_start
            self.logger.info(f"Time communication round: {comm_time}")
            self.comm_times.append(comm_time)
            report = self.validation_round()

            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            for client in self.clients:
                client.set_model(self.model)

            # save model state every 5 communication rounds
            if com_round % 5 == 0:
                self.save_state_dict()

        self.train_time = time.perf_counter() - train_start

        self.client_results = [
            client.get_validation_results() for client in self.clients
        ]
        self.save_results()
        self.save_state_dict()
        return self.results, self.client_results

    def validation_round(self) -> str | dict:
        """
        Perform validation round on global client's model.

        Returns:
            (str | dict): classification report of the current model state.
        """
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for _, batch in enumerate(
                tqdm(self.val_loader, desc="Validation round global model")
            ):
                data = batch["data"].to(self.device)
                labels = batch["label"]
                label_new = np.copy(labels)
                label_new = change_sizes(label_new)

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

    def apply_model_updates(self, model_updates: list[dict]):
        """
        Apply model updates based on the local clients updates.

        Args:
            model_updates (list[dict]): List of all updates of the local clients to consider.
        """
        # parameter aggregation
        self.logger.info("Starting model update aggregation")
        update_aggregation = fed_avg(model_updates)
        self.logger.info("Model update aggregation complete")
        # update the global model
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)
        self.logger.info("Communication round done")

    def sequential_communication_round(self, epochs: int):
        """
        Perform communication round where local clients are trained sequentially.

        Args:
            epochs (int): Amount of epochs to train for.
        """
        if self.device is torch.device("cpu"):
            self.logger.info("Starting communication round on CPU")
        else:
            self.logger.info("Starting communication round on single GPU")
        # here the clients train
        model_updates = []
        for client in self.clients:
            self.logger.info(f"{client.name} training...")
            model_updates.append(client.train_one_round(epochs))
        self.logger.info("All clients done with training")
        self.apply_model_updates(model_updates)

    def parallel_communication_round(self, epochs: int):
        """
        Perform communication round where local clients are trained in parallel.

        Args:
            epochs (int): Amount of epochs to train for.
        """
        self.logger.info(
            f"Starting communication round on ({torch.cuda.device_count()}) GPUs"
        )
        # use mp.Manager to ensure that Locks and Queue are properly shared between processes
        with mp.Manager() as manager:
            model_queue = manager.Queue(
                len(self.train_loaders) + torch.cuda.device_count()
            )
            gpu_locks = [manager.Lock() for _ in range(torch.cuda.device_count())]
            # process large loaders first to minimize idle time
            queue_data = [(epochs, train_loader) for train_loader in self.train_loaders]
            sorted_queue_data = sorted(
                queue_data, key=lambda x: len(x[1]), reverse=True
            )
            # put training data in queue
            for data in sorted_queue_data:
                model_queue.put(data)
            # put termination signal in queue
            for _ in range(torch.cuda.device_count()):
                model_queue.put(None)
            processing_pool = mp.Pool(torch.cuda.device_count())
            model_updates = processing_pool.map(
                parallel_gpu_work,
                [
                    GPUWorker(gpu_id, model_queue, gpu_locks, self.model)
                    for gpu_id in range(torch.cuda.device_count())
                ],
            )
            processing_pool.close()
        self.logger.info("All clients done with training")
        # flatten results list
        model_updates = list(itertools.chain.from_iterable(model_updates))
        self.apply_model_updates(model_updates)

    def save_state_dict(self):
        """
        Save state dict of the current global model.
        """
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        """
        Save all training results.
        """
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)
        res = {
            "global": self.results,
            "clients": self.client_results,
            "train_time": self.train_time,
            "communication_times": self.comm_times,
        }
        torch.save(res, self.results_path)
