import argparse
import warnings
from collections import OrderedDict
import os
# from utils.logging_utils import setup_logger, log_metrics, CommunicationTimer # Keep if still used for other things
import logging # Added
import mqtthandler # Added
import paho.mqtt.client as mqtt # Added
import time # Added for potential use in logging or MQTT connection

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
from sc_manually_quantized import prepare_scrimmage_data, post_training_quantization, compare_model_sizes # Assuming this exists
from models import * # Assuming this exists
from quantize.k_means import KMeansQuantizer # Assuming this exists
from flwr_datasets.partitioner import NaturalIdPartitioner, IidPartitioner # Assuming this exists

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_BASE_TOPIC_CLIENT = "federated_learning/client"

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 10

# --- Define CommunicationTimer and log_metrics if not in utils.logging_utils or if they need MQTT specific adaptations ---
# For demonstration, let's assume a simple CommunicationTimer and log_metrics.
# If these are complex in your utils, you'll need to adapt them accordingly or ensure the logger they use is the MQTT-configured one.

class CommunicationTimer:
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.communication_size = 0 # Placeholder for packet size

    def __enter__(self):
        self.start_time = time.time()
        # Placeholder: you might need to intercept send/receive calls to measure packet size
        # For example, by wrapping socket operations or Flower's transport mechanism if possible.
        self.logger.info(f"Starting communication for {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        # Log communication time and packet size (if measured)
        # The packet size logging would need actual measurement logic
        memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0 # Example memory usage
        self.logger.info(
            f"Finished communication for {self.operation_name}. "
            f"Duration: {duration:.4f}s. "
            f"Packet Size: {self.communication_size} bytes. " # This needs actual implementation
            f"Memory Usage: {memory_usage} bytes."
        )
        # Reset for potential reuse if necessary
        self.communication_size = 0


def log_metrics(logger, metrics_dict):
    for key, value in metrics_dict.items():
        logger.info(f"Metric: {key} = {value}")

# --- Modified/New Logger Setup ---
def setup_mqtt_logger(cid):
    logger = logging.getLogger(f"fl_client_{cid}")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if this function is called multiple times
    if not logger.handlers:
        # Console Handler (optional, for local debugging)
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - Client {cid} - %(message)s"))
        # logger.addHandler(console_handler)

        # MQTT Handler
        mqtt_topic = f"{MQTT_BASE_TOPIC_CLIENT}/{cid}/logs"
        try:
            mqtt_handler = mqtthandler.MQTTHandler(
                host=MQTT_BROKER_HOST,
                topic=mqtt_topic,
                port=MQTT_BROKER_PORT,
                # qos=1, # Optional: Quality of Service
                # retain=False, # Optional: Retain messages
                # client_id=f"fl_client_logger_{cid}" # Optional: custom client ID
            )
            mqtt_handler.setFormatter(logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - Client {cid} - %(message)s"))
            mqtt_handler.setLevel(logging.INFO)
            logger.addHandler(mqtt_handler)
            logger.info(f"MQTT logging initialized for client {cid} on topic {mqtt_topic}")
        except Exception as e:
            logger.error(f"Failed to initialize MQTT handler for client {cid}: {e}")
            # Fallback to basic console logging if MQTT setup fails
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                console_handler_fallback = logging.StreamHandler()
                console_handler_fallback.setFormatter(logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - Client {cid} - Fallback - %(message)s"))
                logger.addHandler(console_handler_fallback)
                logger.info("Fell back to console logging due to MQTT handler initialization error.")
                
    return logger


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


def train(net, trainloader, optimizer, epochs, device, logger): # Added logger
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for epoch_num in range(epochs):
        logger.info(f"Starting epoch {epoch_num + 1}/{epochs}")
        for batch_idx, batch in enumerate(tqdm(trainloader, desc=f"Epoch {epoch_num+1} Training")):
            batch_data = list(batch.values())
            images, labels = batch_data[0], batch_data[1]
            
            if isinstance(labels[0], str):
                unique_labels = list(set(labels))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                labels = [label_to_idx[label] for label in labels]
                labels = torch.tensor(labels)
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0: # Log progress periodically
                 logger.debug(f"Epoch {epoch_num+1}, Batch {batch_idx}: Loss {loss.item():.4f}")
        logger.info(f"Finished epoch {epoch_num + 1}/{epochs}")


def test(net, testloader, device: str = "cpu", logger=None): # Added logger
    """Validate the network on the testing set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            batch_data = list(batch.values())
            images, labels = batch_data[0], batch_data[1]

            if isinstance(labels[0], str):
                unique_labels = list(set(labels))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                labels = [label_to_idx[label] for label in labels]
                labels = torch.tensor(labels)
            
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    if logger:
        logger.info(f"Test Set Evaluation: Loss {loss/len(testloader):.4f}, Accuracy {accuracy:.4f}")
    return loss / len(testloader), accuracy # Return average loss


def prepare_dataset(dataset, non_iid=False):
    """Get dataset and return client partitions and global testset."""
    print("Dataset: ", dataset) # This print will not go to MQTT unless you change it
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
        norm = Normalize((0.1307,), (0.3081,)) # FEMNIST is grayscale, typically uses MNIST normalization
    elif dataset == "cifar10":
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": IidPartitioner(num_partitions=NUM_CLIENTS)})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == "sc2":
        partitions = prepare_scrimmage_data(NUM_CLIENTS) # Assuming this function returns partitions
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    # testsets = [] # testsets per client not used in FlowerClient, only valset
    
    for partition_id in range(NUM_CLIENTS):
        if dataset == "sc2":
            partition = partitions[partition_id] # Assuming partitions is a list of datasets
        else:
            partition = fds.load_partition(partition_id, "train") # Load 'train' split for partitioning
            
        # Split into train (70%) and validation (30%)
        # The original code had a 70/15/15 split, but FlowerClient uses only train and val.
        # If client-side testing is needed, testsets can be prepared and used in evaluate.
        train_val_split = partition.train_test_split(test_size=0.3, seed=42)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"] # 'test' from this split is used as validation

        train_data = train_data.with_transform(apply_transforms)
        val_data = val_data.with_transform(apply_transforms)
        
        trainsets.append(train_data)
        validsets.append(val_data)

    # Global testset - currently not directly used by the client's evaluate function in this setup
    # testset_global = None
    # if dataset != "sc2":
    #     try:
    #         testset_global = fds.load_split("test")
    #         testset_global = testset_global.with_transform(apply_transforms)
    #     except Exception as e:
    #         print(f"Could not load global test set for {dataset}: {e}")

    return trainsets, validsets # Removed testsets per client and global testset from return as they are not directly used

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, dataset, cid):
        self.cid = cid
        self.logger = setup_mqtt_logger(cid) # Use MQTT logger
        
        EMBEDDING_DIM = 2 # Example, adjust as per your SC2 model needs
        HIDDEN_DIM = 100  # Example
        TAGSET_SIZE = 2   # Example
        
        self.trainset = trainset
        self.valset = valset
        
        if dataset in ["mnist", "fashion_mnist"]:
            self.model = LeNet5()
        elif dataset == "femnist":
            self.model = FEMNISTCNN()
        elif dataset == "cifar10":
            self.model = VeryDeepCNN() # Example, ensure this model exists
        elif dataset == "sc2":
            # Ensure BGRUTagger is defined or imported correctly
            self.model = BGRUTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)
        else: # Fallback for other datasets, e.g. mobilenet for CIFAR-10 like structure
            self.logger.warning(f"Dataset {dataset} not explicitly handled, using mobilenet_v3_small. Adjust if needed.")
            self.model = mobilenet_v3_small(num_classes=10) # Assuming 10 classes if not specified
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.logger.info(f"Client {cid} initialized with dataset '{dataset}', model '{type(self.model).__name__}' on device: {self.device}")

    def get_parameters(self, config):
        self.logger.info("get_parameters called")
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        with CommunicationTimer(self.logger, "get_parameters") as timer:
            timer.communication_size = param_bytes 
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        self.logger.info("set_parameters called")
        param_bytes = sum(p.nbytes for p in parameters)
        with CommunicationTimer(self.logger, "set_parameters") as timer:
            timer.communication_size = param_bytes 
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in params_dict} 
            )
            self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.logger.info(f"Starting fit operation with config: {config}")
        with CommunicationTimer(self.logger, "fit_model_transfer_and_train"):
            self.set_parameters(parameters) 
            
            batch_size, epochs = config["batch_size"], config["epochs"]
            trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0) 
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9) 
            
            self.logger.info(f"Training with epochs: {epochs}, batch_size: {batch_size}")
            train(self.model, trainloader, optimizer, epochs=epochs, device=self.device, logger=self.logger)
            
            metrics = {
                "training_epochs": epochs,
                "batch_size": batch_size,
                "dataset_size": len(trainloader.dataset)
            }
            log_metrics(self.logger, metrics) 
            self.logger.info("Fit operation completed.")
            
            return self.get_parameters({}), len(trainloader.dataset), {}


    def evaluate(self, parameters, config):
        self.logger.info(f"Starting evaluate operation with config: {config}")
        with CommunicationTimer(self.logger, "evaluate_model_transfer_and_test"): 
            self.set_parameters(parameters)
            
            valloader = DataLoader(self.valset, batch_size=64, num_workers=0) 
            loss, accuracy = test(self.model, valloader, device=self.device, logger=self.logger)
            
            metrics = {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
                "validation_dataset_size": len(valloader.dataset)
            }
            log_metrics(self.logger, metrics)
            
            self.logger.info("Evaluate operation completed.")
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
    # Basic logging for main function before client-specific logger is up
    # This will use root logger, which won't go to MQTT unless root is configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Client arguments: {args}")


    if args.cid >= NUM_CLIENTS:
        logging.error(f"Client ID {args.cid} is out of range (NUM_CLIENTS={NUM_CLIENTS}).")
        return

    dataset_name = args.dataset.lower()
    logging.info(f"Preparing dataset: {dataset_name}, Non-IID: {args.non_iid}")
    trainsets, valsets = prepare_dataset(dataset_name, non_iid=args.non_iid)

    client_instance = FlowerClient(
        trainset=trainsets[args.cid],
        valset=valsets[args.cid],
        dataset=dataset_name,
        cid=args.cid
    )
    
    logging.info(f"Starting Flower client {args.cid} for server {args.server_address}")
    fl.client.start_client(
        server_address=args.server_address,
        client=client_instance.to_client(),
    )
    logging.info(f"Flower client {args.cid} finished.")

if __name__ == "__main__":
    main()