import argparse
import warnings
from collections import OrderedDict
import os
from utils.logging_utils import setup_logger, log_metrics, CommunicationTimer

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from sc_manually_quantized import prepare_scrimmage_data, post_training_quantization, compare_model_sizes
from models import *
from quantize.k_means import KMeansQuantizer
from flwr_datasets.partitioner import NaturalIdPartitioner, IidPartitioner

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="Dataset to use. Options: SC2, cifar10, mnist, femnist",
)
parser.add_argument(
    "--non_iid",
    action="store_true",
    default=False,
    help="Use non-IID partitioning for the dataset",
)


warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 10

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            # Convert string labels to numeric indices
            if isinstance(labels[0], str):
                # Create a mapping of unique labels to indices
                unique_labels = list(set(labels))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                # Convert labels to indices
                labels = [label_to_idx[label] for label in labels]
                # Convert to tensor
                labels = torch.tensor(labels)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str = "cpu"):
    """Validate the network on the testing set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in tqdm(testloader):
            # Handle dictionary format from FederatedDataset
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            # print(labels)
            if isinstance(labels[0], str):
                # Create a mapping of unique labels to indices
                unique_labels = list(set(labels))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                # Convert labels to indices
                labels = [label_to_idx[label] for label in labels]
                # Convert to tensor
                labels = torch.tensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def prepare_dataset(dataset, non_iid=False):
    """Get dataset and return client partitions and global testset."""
    print("Dataset: ", dataset)
    if dataset == "mnist":
        fds = FederatedDataset(dataset="mnist", partitioners={"train": IidPartitioner(num_partitions=NUM_CLIENTS)})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    elif dataset == "femnist":
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
        )
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    elif dataset == "cifar10":
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": IidPartitioner(num_partitions=NUM_CLIENTS)})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == "sc2":
        partitions = prepare_scrimmage_data(NUM_CLIENTS)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    testsets = []
    
    for partition_id in range(NUM_CLIENTS):
        if dataset == "sc2":
            partition = partitions[partition_id]
        else:
            partition = fds.load_partition(partition_id)
            
        # Split into train (70%), validation (15%), and test (15%)
        if dataset == "femnist":
            # For FEMNIST, we need to handle the writer_id-based partitioning
            train_val = partition.train_test_split(test_size=0.3, seed=42)
            val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)
            
            train = train_val["train"]
            val = val_test["train"]
            test = val_test["test"]
        else:
            # For other datasets, use standard split
            train_val = partition.train_test_split(test_size=0.3, seed=42)
            val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)
            
            train = train_val["train"]
            val = val_test["train"]
            test = val_test["test"]
        
        # Apply transforms
        train = train.with_transform(apply_transforms)
        val = val.with_transform(apply_transforms)
        test = test.with_transform(apply_transforms)
        
        trainsets.append(train)
        validsets.append(val)
        testsets.append(test)

    if dataset != "sc2":
        # Load global test set if available
        try:
            testset = fds.load_split("test")
            testset = testset.with_transform(apply_transforms)
        except:
            testset = None
    else:
        testset = None

    return trainsets, validsets, testsets, testset


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset, dataset, cid):
        EMBEDDING_DIM = 2
        HIDDEN_DIM = 100
        TAGSET_SIZE = 2
        NUM_EPOCHS = 1
        TRAIN_BATCH_SIZE = 1024
        VAL_BATCH_SIZE = 128
        self.trainset = trainset
        self.valset = valset
        self.cid = cid
        self.logger = setup_logger(cid)
        
        # Instantiate model
        if dataset in ["mnist", "fashion_mnist"]:
            self.model = LeNet5()
        elif dataset == "femnist":
            self.model = FEMNISTCNN()
        elif dataset == "cifar10":
            self.model = VeryDeepCNN()
        elif dataset == "sc2":
            self.model = BGRUTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)
        else:
            self.model = mobilenet_v3_small(num_classes=10)
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.bitwidths = [8, 4, 2, 1]
        
        self.logger.info(f"Client {cid} initialized with device: {self.device}")

    def set_parameters(self, params):
        with CommunicationTimer(self.logger, "set_parameters"):
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict(
                {
                    k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                    for k, v in params_dict
                }
            )
            self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        with CommunicationTimer(self.logger, "get_parameters"):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.logger.info("Starting fit operation")
        with CommunicationTimer(self.logger, "fit"):
            self.set_parameters(parameters)
            batch, epochs = config["batch_size"], config["epochs"]
            trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            
            train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
            
            # Log metrics after training
            metrics = {
                "training_epochs": epochs,
                "batch_size": batch,
                "dataset_size": len(trainloader.dataset)
            }
            log_metrics(self.logger, metrics)
            
            return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.logger.info("Starting evaluate operation")
        with CommunicationTimer(self.logger, "evaluate"):
            self.set_parameters(parameters)
            valloader = DataLoader(self.valset, batch_size=64)
            loss, accuracy = test(self.model, valloader, device=self.device)
            
            # Log evaluation metrics
            metrics = {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
                "validation_dataset_size": len(valloader.dataset)
            }
            log_metrics(self.logger, metrics)
            
            # Save and quantize model
            os.makedirs("models", exist_ok=True)
            original_model_path = "models/original_model.pt"
            torch.save(self.model.state_dict(), original_model_path)
            
            saved_model = type(self.model)()
            saved_model.load_state_dict(torch.load(original_model_path))
            quantized_model = post_training_quantization(saved_model, valloader, original_model_path)
            torch.save(quantized_model.state_dict(), "models/quantized_model.pt")
            
            compare_model_sizes(self.model, quantized_model)
            
            return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    dataset = args.dataset.lower()
    # Download dataset and partition it
    trainsets, valsets, testsets, testset = prepare_dataset(dataset, non_iid=args.non_iid)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid],
            valset=valsets[args.cid],
            dataset=dataset,
            cid=args.cid
        ).to_client(),
    )


if __name__ == "__main__":
    main()
