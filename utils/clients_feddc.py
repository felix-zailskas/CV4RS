import copy
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.poolformer import PoolFormer
from utils.pytorch_datasets import Ben19Dataset
from utils.pytorch_models import ResNet50
from utils.pytorch_utils import (
    get_classification_report,
    init_results,
    print_micro_macro,
    start_cuda,
    update_results,
)

model_args = {"name": "ResNet50", "channels": 10, "num_cls": 19}
model_func = ResNet50


def change_sizes(labels):
    new_labels = np.zeros((len(labels[0]), 19))
    for i in range(len(labels[0])):  # 128
        for j in range(len(labels)):  # 19
            new_labels[i, j] = int(labels[j][i])
    return new_labels


def train_model_FedDC(
    model,
    data_loader,
    device,
    epochs,
    alpha,
    local_update_last,
    global_update_last,
    global_model_param,
    hist_i,
):
    state_update_diff = torch.tensor(
        -local_update_last + global_update_last, dtype=torch.float32, device=device
    )

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    model.train()
    model.to(device)

    n_par = get_mdl_params([model_func(**model_args)]).shape[1]

    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        for idx, batch in enumerate(tqdm(data_loader, desc="training")):
            data, labels, index = batch["data"], batch["label"], batch["index"]
            data = data.to(device)
            label_new = np.copy(labels)
            label_new = change_sizes(label_new)
            label_new = torch.from_numpy(label_new).to(device)

            y_pred = model(data)

            loss_f_i = loss_fn(y_pred, label_new)
            loss_f_i = loss_f_i / len(batch["label"])

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = (
                alpha
                / 2
                * torch.sum(
                    (local_parameter - (global_model_param - hist_i))
                    * (local_parameter - (global_model_param - hist_i))
                )
            )
            loss_cg = torch.sum(local_parameter * state_update_diff)

            loss = loss_f_i + loss_cp + loss_cg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=10
            )  # Clip gradients to prevent exploding
            optimizer.step()

    # freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


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


class GlobalClientFedDC:
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
    ) -> None:
        self.model = model
        self.device = (
            torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.lmdb_path = lmdb_path
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)

        # set up clients
        self.n_clients = len(csv_paths)
        self.clients = [csv_path for csv_path in csv_paths]
        self.client_data_lengths = []
        for client in self.clients:
            with open(client, "r") as fh:
                csv_reader = csv.reader(fh)
                self.client_data_lengths.append(len(list(csv_reader)))
        self.client_data_lengths = np.asarray(self.client_data_lengths)
        self.weight_list = (
            self.client_data_lengths / np.sum(self.client_data_lengths) * self.n_clients
        )

        self.data_loaders = [
            DataLoader(
                Ben19Dataset(lmdb_path, csv_path),
                batch_size=128,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
            for csv_path in csv_paths
        ]

        # set up model parameters
        self.alpha_coef = 0.1
        self.init_model = model_func(**model_args)
        self.n_par = len(get_mdl_params([model_func(**model_args)])[0])
        self.parameter_drifts = np.zeros((self.n_clients, self.n_par)).astype("float32")
        self.init_par_list = get_mdl_params([self.init_model])[0]
        # dim = (#clients, #params)
        self.clnt_params_list = np.ones(self.n_clients).astype("float32").reshape(
            -1, 1
        ) * self.init_par_list.reshape(1, -1)
        self.clnt_models = list(range(self.n_clients))
        # includes global model
        self.state_gradient_diffs = np.zeros((self.n_clients + 1, self.n_par)).astype(
            "float32"
        )

        """
        self.global_state_gradient_diff = torch.zeros(
            self.n_par, dtype=torch.float32, device=self.device
        )
        self.delta_g_sum = torch.zeros(self.n_par, device=self.device)
        # self.global_model_param = get_mdl_params(self.model, self.device, self.n_par)
        self.clients = [
            FLCLient(
                copy.deepcopy(self.model),
                self,
                lmdb_path,
                val_path,
                csv_path,
                num_classes=num_classes,
                dataset_filter=dataset_filter,
                device=self.device,
            )
            for csv_path in csv_paths
        ]
        self.total_len_dataset = sum([cl.len_dataset for cl in self.clients])
        for cl in self.clients:
            cl.set_total_len_dataset(self.total_len_dataset)
        """
        # old code
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
        self.train_time = None

        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if state_dict_path is None:
            if isinstance(model, ConvMixer):
                self.state_dict_path = f"checkpoints/global_convmixer_{dt}.pkl"
            elif isinstance(model, MlpMixer):
                self.state_dict_path = f"checkpoints/global_mlpmixer_{dt}.pkl"
            elif isinstance(model, PoolFormer):
                self.state_dict_path = f"checkpoints/global_poolformer_{dt}.pkl"
            elif isinstance(model, ResNet50):
                self.state_dict_path = f"checkpoints/global_resnet50_{dt}.pkl"

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f"results/convmixer_results_{dt}.pkl"
            elif isinstance(model, MlpMixer):
                self.results_path = f"results/mlpmixer_results_{dt}.pkl"
            elif isinstance(model, PoolFormer):
                self.results_path = f"results/poolformer_results_{dt}.pkl"
            elif isinstance(model, ResNet50):
                self.results_path = f"results/resnet50_results_{dt}.pkl"

    def train(self, communication_rounds: int, epochs: int):
        # set up models
        self.avg_model = model_func(**model_args).to(self.device)
        self.avg_model.load_state_dict(
            copy.deepcopy(dict(self.init_model.named_parameters())), strict=False
        )

        # self.all_model = model_func(**model_args).to(self.device)
        # self.all_model.load_state_dict(copy.deepcopy(dict(self.init_model.named_parameters())), strict=False)

        self.cur_cld_model = model_func(**model_args).to(self.device)
        self.cur_cld_model.load_state_dict(
            copy.deepcopy(dict(self.init_model.named_parameters())), strict=False
        )
        self.cld_mdl_param = get_mdl_params([self.cur_cld_model], self.n_par)[0]

        self.comm_times = []
        start = time.perf_counter()
        # start communication rounds
        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            # communication round
            comm_start = time.perf_counter()
            self.communication_round(epochs)
            comm_time = time.perf_counter() - comm_start
            print(f"Time communication round: {comm_time}")
            self.comm_times.append(comm_time)
            print(torch.cuda.memory_summary())
            report = self.validation_round()

            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            for client in self.clients:
                client.set_model(self.model)

            if com_round % 5 == 0:
                self.save_state_dict()
                self.client_results = [
                    client.get_validation_results() for client in self.clients
                ]
                # self.save_results()

        self.train_time = time.perf_counter() - start

        self.client_results = [
            client.get_validation_results() for client in self.clients
        ]
        self.save_results()
        self.save_state_dict()
        return self.results, self.client_results

    def change_sizes(self, labels):
        new_labels = np.zeros((len(labels[0]), 19))
        for i in range(len(labels[0])):  # 128
            for j in range(len(labels)):  # 19
                new_labels[i, j] = int(labels[j][i])
        return new_labels

    def validation_round(self):
        self.cur_cld_model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch["data"].to(self.device)
                labels = batch["label"]
                label_new = np.copy(labels)
                label_new = change_sizes(label_new)

                logits = self.cur_cld_model(data)
                print(
                    f"{batch_idx} : MODEL PREDICTION NAN: {torch.isnan(logits).any()}"
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                print(
                    f"{batch_idx} : SIGMOID FUNCTION NAN: {torch.isnan(logits).any()}"
                )

                predicted_probs += list(probs)

                y_true += list(label_new)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)

        y_true = np.asarray(y_true)
        report = get_classification_report(
            y_true, y_predicted, predicted_probs, self.dataset_filter
        )
        self.cur_cld_model.train()
        return report

    def communication_round(self, epochs: int):
        global_mdl = torch.tensor(
            self.cld_mdl_param, dtype=torch.float32, device=self.device
        )
        del self.clnt_models
        self.clnt_models = list(range(self.n_clients))
        delta_g_sum = np.zeros(self.n_par)

        for clnt_i, clnt in enumerate(self.clients):
            print(f"Train client {clnt}")
            # load current global model into client
            self.clnt_models[clnt_i] = model_func(**model_args).to(self.device)
            model = self.clnt_models[clnt_i]
            model.load_state_dict(
                copy.deepcopy(dict(self.cur_cld_model.named_parameters())), strict=False
            )
            # make sure to collect the gradients
            for params in model.parameters():
                params.requires_grad = True
            local_update_last = self.state_gradient_diffs[clnt_i]
            # global model at last pos
            global_update_last = (
                self.state_gradient_diffs[-1] / self.weight_list[clnt_i]
            )
            alpha = self.alpha_coef / self.weight_list[clnt_i]
            hist_i = torch.tensor(
                self.parameter_drifts[clnt_i], dtype=torch.float32, device=self.device
            )
            # train client
            self.clnt_models[clnt_i] = train_model_FedDC(
                model,
                self.data_loaders[clnt_i],
                self.device,
                epochs,
                alpha,
                local_update_last,
                global_update_last,
                global_mdl,
                hist_i,
            )

            curr_model_par = get_mdl_params([self.clnt_models[clnt_i]])
            delta_param_curr = curr_model_par - self.cld_mdl_param
            self.parameter_drifts[clnt_i] += delta_param_curr.reshape(
                delta_param_curr.shape[1],
            )
            n_minibatch = (
                np.ceil(self.client_data_lengths[clnt_i] / 128) * epochs
            ).astype(np.int64)
            beta = 1 / n_minibatch / 0.001

            state_g = local_update_last - global_update_last + beta * (delta_param_curr)
            delta_g_cur = (
                state_g - self.state_gradient_diffs[clnt_i]
            ) * self.weight_list[clnt_i]
            delta_g_sum += delta_g_cur.reshape(
                delta_g_cur.shape[1],
            )
            self.state_gradient_diffs[clnt_i] = state_g
            self.clnt_params_list[clnt_i] = curr_model_par.reshape(
                curr_model_par.shape[1],
            )

        avg_mdl_param = np.mean(self.clnt_params_list, axis=0)
        delta_g_cur = 1 / self.n_clients * delta_g_sum
        self.state_gradient_diffs[-1] += delta_g_cur.reshape(
            delta_g_cur.shape[1],
        )

        self.cld_mdl_param = avg_mdl_param + np.mean(self.parameter_drifts, axis=0)
        self.avg_model = set_client_from_params(
            model_func(**model_args),
            np.mean(self.clnt_params_list, axis=0),
            self.device,
        )
        self.cur_cld_model = set_client_from_params(
            model_func(**model_args).to(self.device),
            np.mean(self.clnt_params_list, axis=0),
            self.device,
        )

    def save_state_dict(self):
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)
        res = {
            "global": self.results,
            "clients": self.client_results,
            "train_time": self.train_time,
            "communication_times": self.comm_times,
        }
        torch.save(res, self.results_path)


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype("float32")
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx : idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def set_client_from_params(mdl, params, device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(
            torch.tensor(params[idx : idx + length].reshape(weights.shape)).to(device)
        )
        idx += length

    mdl.load_state_dict(dict_param, strict=False)
    return mdl
