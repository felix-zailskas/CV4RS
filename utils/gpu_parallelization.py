import copy
import datetime
import numpy as np
import torch
from tqdm import tqdm

def parallel_gpu_work(gpu_worker):
    results = []
    while True:
        # Dequeue a model input to train
        model_input = gpu_worker.model_queue.get()

        # Check if the model input is a sentinel value indicating the end of training
        if model_input is None:
            print(f"GPU {gpu_worker.gpu_id}: No more models to train. Exiting.")
            break

        # Train the model and get the result
        with gpu_worker.gpu_lock:
            epochs, train_loader = model_input
            gpu_worker.train_loader = train_loader
            result = gpu_worker.train_one_round(epochs)

            # Append the result to the local results list
            print(f"GPU {gpu_worker.gpu_id}: Appended result to the list")
            results.append(result)
    return results

class GPUWorker:
    def __init__(self, gpu_id, model_queue, gpu_locks, model) -> None:
        self.gpu_id = gpu_id
        self.model_queue = model_queue
        self.gpu_lock = gpu_locks[gpu_id]
        # Create a new instance of the model on the GPU
        self.model = copy.deepcopy(model).cuda(self.gpu_id)
    
    def train_one_round(
        self, epochs: int
    ):
        state_before = copy.deepcopy(self.model.state_dict())
        self.global_model = copy.deepcopy(
            self.model
        )  # save current model as global model
        print(f"Putting model {self} onto {torch.device(self.gpu_id)}")
        self.model.to(torch.device(self.gpu_id))
        self.global_model.to(torch.device(self.gpu_id))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))

            self.train_epoch(torch.device(self.gpu_id))
        print("Training done!")

        state_after = self.model.state_dict()
        self.previous_model = copy.deepcopy(
            self.model
        )  # save previous model for next iteration

        model_update = {}
        for key, value_before in state_before.items():
            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(
                torch.DoubleTensor
            )
            model_update[key] = diff

        return model_update

    def change_sizes(self, labels):
        new_labels = np.zeros((len(labels[0]), 19))
        for i in range(len(labels[0])):  # 128
            for j in range(len(labels)):  # 19
                new_labels[i, j] = int(labels[j][i])
        return new_labels

    def train_epoch(self, training_device):
        self.model.train()
        for idx, batch in enumerate(
            tqdm(
                self.train_loader,
                desc=f"training {self} on device {training_device if training_device is not None else self.device}",
            )
        ):
            data, labels, index = batch["data"], batch["label"], batch["index"]
            data = data.to(training_device)
            label_new = np.copy(labels)
            label_new = self.change_sizes(label_new)
            label_new = torch.from_numpy(label_new).to(training_device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()