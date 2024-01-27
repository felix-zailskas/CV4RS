import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path 

from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50

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
        n_clients,
        batch_size: int = 128,
        num_workers: int = 0,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device('cpu'),
        dataset_filter: str = "serbia",
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.n_clients = n_clients
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)
        self.dataset = Ben19Dataset(lmdb_path, csv_path)
        self.len_dataset = len(self.dataset)
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
        self.alpha = 0.1
        # TODO receive n_par from global model
        n_par = 0
        for param in self.model.parameters():
            n_par += len(param.data.reshape(-1))
        self.state_gadient_diff = np.zeros((n_par)).astype('float32') 

        self.parameter_drift =  np.zeros((n_par)).astype('float32') 

    def set_model(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)

    def set_total_len_daatset(self, total_len_dataset):
        self.total_len_dataset = total_len_dataset

    def train_one_round(self, epochs: int, global_state_gradient_diff: torch.Tensor, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())
        
        # TODO change everything to either state_dict or parameters
        global_parameter = None
        for param in self.model.parameters():
                if not isinstance(global_parameter, torch.Tensor):
                # Initially nothing to concatenate
                    global_parameter = param.reshape(-1)
                else:
                    global_parameter = torch.cat((global_parameter, param.reshape(-1)), 0)
        self.local_update_last = self.state_gadient_diff # delta theta_i
        dataset_weight = self.len_dataset / self.total_len_dataset * self.n_clients
        self.global_update_last = global_state_gradient_diff/dataset_weight #delta theta
        self.state_update_diff = torch.tensor(-self.local_update_last+ self.global_update_last,  dtype=torch.float32, device=self.device)
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)

            self.train_epoch(global_parameter)
        
        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)

        state_after = self.model.state_dict()

        for param in self.model.parameters():
            if not isinstance(local_parameter, torch.Tensor):
            # Initially nothing to concatenate
                local_parameter = param.reshape(-1)
            else:
                local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

        delta_param_curr = local_parameter - global_parameter
        self.parameter_drift += delta_param_curr
        # n_minibatch = np.ceil(self.len_dataset/self.batch_size) * epochs
        # beta = 1/n_minibatch/self.optimizer_kwargs['lr']
        # state_g = self.local_update_last - self.global_update_last + beta * (-delta_param_curr)
        # delta_g_cur = (state_g - self.state_gadient_diff)*dataset_weight 
        # delta_g_sum += delta_g_cur
        # self.state_gadient_diff = state_g 
        # clnt_params_list[clnt] = curr_model_par 
        
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
    
    def train_epoch(self, global_parameter):
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data, labels, index = batch["data"], batch["label"], batch["index"]
            data = data.cuda()
            label_new=np.copy(labels)
            label_new=self.change_sizes(label_new)
            label_new = torch.from_numpy(label_new).cuda()
            self.optimizer.zero_grad()
            local_parameter = None
            for param in self.model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
            logits = self.model(data)
            loss_cp = self.alpha/2 * torch.sum((local_parameter - (global_parameter - self.parameter_drift))*(local_parameter - (global_parameter - self.parameter_drift)))
            loss_cg = torch.sum(local_parameter * self.state_update_diff) 
            

            loss = self.criterion(logits, label_new) + loss_cp + loss_cg
            # loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()
    
            #     for clnt in selected_clnts:
            #     print('---- Training client %d' %clnt)
            #     trn_x = clnt_x[clnt]
            #     trn_y = clnt_y[clnt]
            #     clnt_models[clnt] = model_func().to(device)
            #     model = clnt_models[clnt]
            #     model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
            #     for params in model.parameters():
            #         params.requires_grad = True
            #     local_update_last = state_gadient_diffs[clnt] # delta theta_i
            #     global_update_last = state_gadient_diffs[-1]/weight_list[clnt] #delta theta
            #     alpha = alpha_coef / weight_list[clnt] 
            #     hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device) #h_i
            #     clnt_models[clnt] = train_model_FedDC(model, model_func, alpha,local_update_last, global_update_last,global_mdl, hist_i, 
            #                                          trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), 
            #                                          batch_size, epoch, print_per, weight_decay, data_obj.dataset, sch_step, sch_gamma)


            #     curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
            #     delta_param_curr = curr_model_par-cld_mdl_param
            #     parameter_drifts[clnt] += delta_param_curr 
            #     beta = 1/n_minibatch/learning_rate
                
            #     state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
            #     delta_g_cur = (state_g - state_gadient_diffs[clnt])*weight_list[clnt] 
            #     delta_g_sum += delta_g_cur
            #     state_gadient_diffs[clnt] = state_g 
            #     clnt_params_list[clnt] = curr_model_par 
                

                        
            # avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis = 0)
            # delta_g_cur = 1 / n_clnt * delta_g_sum  
            # state_gadient_diffs[-1] += delta_g_cur  
            
            # cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)
            
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
        results_path: str = None
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)
        n_clients = len(csv_paths)
        self.clients = [
            FLCLient(copy.deepcopy(self.model), lmdb_path, val_path, csv_path, n_clients, num_classes=num_classes, dataset_filter=dataset_filter, device=self.device)
            for csv_path in csv_paths
        ]
        self.total_len_dataset = sum([cl.len_dataset for cl in self.clients])
        for cl in self.clients:
            cl.set_total_len_daatset(self.total_len_dataset)
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
        n_par = 0
        for param in self.model.parameters():
            n_par += len(param.data.reshape(-1))
        self.global_state_gradient_diff = np.zeros((n_par)).astype('float32') 

        
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if state_dict_path is None:
            if isinstance(model, ConvMixer):
                self.state_dict_path = f'checkpoints/global_convmixer_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.state_dict_path = f'checkpoints/global_mlpmixer_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.state_dict_path = f'checkpoints/global_poolformer_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.state_dict_path = f'checkpoints/global_resnet50_{dt}.pkl'

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f'results/convmixer_results_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.results_path = f'results/mlpmixer_results_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.results_path = f'results/poolformer_results_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.results_path = f'results/resnet50_results_{dt}.pkl'

    def train(self, communication_rounds: int, epochs: int):
        self.comm_times = []
        start = time.perf_counter()
        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            # communication round
            comm_start = time.perf_counter()
            self.communication_round(epochs)
            comm_time = time.perf_counter() - comm_start
            print(f"Time communication round: {comm_time}")
            self.comm_times.append(comm_time)
            
            report = self.validation_round()

            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            for client in self.clients:
                client.set_model(self.model)
        
            if com_round % 5 == 0:
                self.save_state_dict()
                self.client_results = [client.get_validation_results() for client in self.clients]
                # self.save_results()
        
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

    def communication_round(self, epochs: int):
        # here the clients train
        # TODO: could be parallelized
        model_updates = [client.train_one_round(epochs, self.global_state_gradient_diff) for client in self.clients]

        # parameter aggregation
        update_aggregation = self.aggregator.fed_avg(model_updates)

        # update the global model
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)

    def save_state_dict(self):
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)  
        res = {'global':self.results, 'clients':self.client_results, 'train_time': self.train_time, 'communication_times':self.comm_times}
        torch.save(res, self.results_path)
