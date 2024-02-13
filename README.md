# Federated Learning Simulation for Sentinel-2 Data from the BigEarthNet Dataset

## Supported Models

The following models are available in the current project version:

<!-- TODO: add paper links for each model type -->

- ResNet
- MLP-Mixer
- ConvMixer
- PoolFormer

However any model that extends the `torch.nn.Module` class should be compatible with the code.

## Execution

The `train.py` script found at the root of this project starts a training run using one of the supported models and data set splits. It initializes one global model and one local client for each `.csv` file in the `data/scenario*/` directories. At the start of the script the following hyperparameters are set:

```
LOCAL_EPOCHS = 20  # amount of epochs each client trains for locally
GLOBAL_COMMUNICATION_ROUNDS = 40  # amount of communication rounds the global model aggregates the local results
NUM_CHANNELS = 10
NUM_CLASSES = 19
```

These should be adjusted to the needed use case.

An example command for execution would be

```
CUDA_VISIBLE_DEVICES=0 numactl -C 0-5 python3 train.py -DS 1 --model resnet --algo fedavg
```

### CPU Usage Limit

To prevent the training script to take over too much of the CPU workload and essentially crash the system use the `numactl -C 0-5` command when executing the training script.

### GPU Parallelization

The current project version supports training multiple simulated local clients on different GPUs at the same time. The training script will use all available `cuda` devices it can find. To set the GPUs to use during training use the `CUDA_VISIBLE_DEVICES` environment variable. Parallelization is done by creating a queue of local clients that need to be trained. Then a `gpu_parallelization.GPUWorker` object is initialized for each available GPU which contains this queue. Using `multiprocessing.Lock` objects it is ensured that only one model is trained on each GPU at the same time. The results of this training are returned to the global model and then aggregated normaly.

Note that in our tests the usage of multiple GPUs for the same training run did not result in a significant decrease in training time. There seems to be some sort of overhead to schedule the different training processes and accessing the same training data in our server. Hence the multiprocessing logic is only actiavted when multiple GPUs are made available to the training script. If you do not want to use it make sure to only make one GPU visible to the training script. Furthermore, note that all GPU workers log only to stdout.

To activate training with a single GPU training make that GPU visible (Recommended):

```
CUDA_VISIBLE_DEVICES=0
```

To activate multiple GPU training make multiple GPUs visible:

```
CUDA_VISIBLE_DEVICES=0,1
```

To deactivate the usage of GPUs make none visible:

```
CUDA_VISIBLE_DEVICES=""
```

### Dataset Split Scenarios

The current version supports three data set splits representing different amounts of non-IID data distribution. The needed `.csv` files can be found under `data/scenario*`.

- Scenario 1: The input data of each country is split randomly across all local clients (low non-IID).
- Scenario 2: The input data of each country is split by the country. So each local client contains data for one country (medium non-IID).
- Scenario 3: The input data of each country is split by the country and season. So each local client contains data for one country in one season (high non-IID).

This input can be controlled by the execution flag `-DS` it expects an input of [1, 2, 3].

### Selected Model

The current version supports four model types as described above. Which one should be used can be controlled by the execution flag `--model` it expects an input of [resnet, mlpmixer, convmixer, poolformer].

### Averaging Algorithm

The current version supports the FedAvg and FedDC averaging algorithms. Which one should be used can be controlled by the execution flag `--algo` it expects an input of [fedavg, feddc]. The default is FedAvg. Note that the parallelized GPU logic only supports FedAvg.

## Logs

Every training run produces log files in which the user can track the training progress. These will be saved under `logs/{scenario}/{model_type}/{averaging_algorithm}/epochs({LOCAL_EPOCHS})_comrounds({GLOBAL_COMMUNICATION_ROUNDS})_{current_time}`. There will be one global log file which tracks the progress of the global model as well as some meta information about the clients training, and one additional log file for each local client which tracks more detailed information about the local training progress. In the case of actiavted GPU parallelization the log files for the local clients will be replaced by a logs of the GPUWorkers to stdout only.
