import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import horovod.torch as hvd

# Assuming JAX code here, but for PyTorch to JAX conversion, adjustments needed.

# Data Creators
def get_tcmf_data_loader(config):
    from bigdl.chronos.model.tcmf.local_model import TCMFDataLoader
    return TCMFDataLoader(
        ray.get(config["Ymat_id"]),
        config["vbsize"],
        config["hbsize"],
        config["end_index"],
        config["val_len"],
        ray.get(config["covariates_id"]),
        config["Ycov_id"],
    )

class TcmfTrainDatasetDist(Dataset):
    def __init__(self, config):
        self.tcmf_data_loader = get_tcmf_data_loader(config)
        self.last_epoch = 0

    def next_batch(self):
        inp, out, _, _ = self.tcmf_data_loader.next_batch()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            bs = inp.shape[0] // world_size
            inp_parts = torch.split(inp, bs)
            out_parts = torch.split(out, bs)
            sid = dist.get_rank()
            return inp_parts[sid], out_parts[sid]
        return inp, out

class TcmfTrainDatasetHorovod(Dataset):
    def __init__(self, config):
        self.tcmf_data_loader = get_tcmf_data_loader(config)
        self.last_epoch = 0

    def next_batch(self):
        inp, out, _, _ = self.tcmf_data_loader.next_batch()
        try:
            hvd_size = hvd.size()
            bs = inp.shape[0] // hvd_size
            inp_parts = torch.split(inp, bs)
            out_parts = torch.split(out, bs)
            sid = hvd.rank()
            return inp_parts[sid], out_parts[sid]
        return inp, out

class TcmfValDataset(Dataset):
    def __init__(self, config):
        self.tcmf_data_loader = get_tcmf_data_loader(config)

    def test(self):
        inp, out = self.tcmf_data_loader.supply_test()
        return inp, out

# Loss and Optimizer
def tcmf_loss(out, target):
    return nn.L1Loss()(out, target)

def loss_creator(config):
    return tcmf_loss

def optimizer_creator(model):
    return optim.Adam(model.parameters(), lr=config["lr"])

# Model
def model_creator(config):
    return TemporalConvNet(
        num_inputs=config["num_inputs"],
        num_channels=config["config"]["num_channels"],
        kernel_size=config["config"]["kernel_size"],
        dropout=config["config"]["dropout"],
        init=True,
    )

# Training
def train(config, workers_per_node, epochs):
    from bigdl.orca.learn.pytorch import Estimator
    estimator = Estimator.from_pytorch(
        model=model_creator(config),
        optimizer=optimizer_creator(model_creator(config)),
        loss=loss_creator(config),
        workers_per_node=workers_per_node,
        backend="horovod",
    )
    #... (rest of training logic)
    return estimator.get_model(), estimator.metrics["val_loss"]