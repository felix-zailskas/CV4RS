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
VISIBLE_CUDA_DEVICES=0,1 python3 train.py -DS 1 --model resnet
```

### GPU Parallelization

The current project version supports training multiple simulated local clients on different GPUs at the same time. The training script will use all available `cuda` devices it can find. To limit the used GPUs use the following command when executing the training script

```
VISIBLE_CUDA_DEVICES=0,1
```

### Dataset Split Scenarios

The current version supports three data set splits representing different amounts of non-IID data distribution. The needed `.csv` files can be found under `data/scenario*`.

- Scenario 1: The input data of each country is split randomly across all local clients (low non-IID).
- Scenario 2: The input data of each country is split by the country. So each local client contains data for one country (medium non-IID).
- Scenario 3: The input data of each country is split by the country and season. So each local client contains data for one country in one season (high non-IID).

This input can be controlled by the execution flag `-DS` it expects an input of [1, 2, 3].

### Selected Model

The current version supports four model types as described above. Which one should be used can be controlled by the execution flag `--model` it expects an input of [resnet, mlpmixer, convmixer, poolformer].

## Logs

Every training run produces log files in which the user can track the training progress. These will be saved under `logs/{scenario}_{model_type}_{current_date}/`. There will be one global log file which tracks the progress of the global model as well as some meta information about the clients training, and one additional log file for each local client which tracks more detailed information about the local training progress.
