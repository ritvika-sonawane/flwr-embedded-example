# python client_pytorch.py --cid 0 --server_address="0.0.0.0:8080" --dataset mnist
import argparse
import warnings
from collections import OrderedDict, namedtuple
import os
import logging
import json
import mqtthandler
# import paho.mqtt.client as mqtt
import time
import flwr as fl
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from models import *
from quantize.k_means import KMeansQuantizer, Codebook
from flwr_datasets.partitioner import NaturalIdPartitioner, IidPartitioner
import numpy as np

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_BASE_TOPIC_CLIENT = "federated_learning/client"

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 10
QUANTIZATION_BITS = 8  # Fixed to 8 bits

# --- Custom JSON Formatter for MQTT Logging ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "client_id": getattr(record, 'client_id', None),
            "operation": getattr(record, 'operation', None),
            "metrics": getattr(record, 'metrics', None),
            "duration": getattr(record, 'duration', None),
            "packet_size": getattr(record, 'packet_size', None),
            "memory_usage": getattr(record, 'memory_usage', None)
        }
        # Remove None values to keep JSON clean
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        return json.dumps(log_entry)

# --- Define CommunicationTimer with JSON logging ---
class CommunicationTimer:
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.communication_size = 0

    def __enter__(self):
        self.start_time = time.time()
        # Log start of communication
        extra = {
            'operation': self.operation_name,
            'event': 'communication_start'
        }
        self.logger.info(f"Starting communication for {self.operation_name}", extra=extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        import psutil
        memory_usage = psutil.Process().memory_info().rss
        
        # Log end of communication with metrics
        extra = {
            'operation': self.operation_name,
            'event': 'communication_end',
            'duration': duration,
            'packet_size': self.communication_size,
            'memory_usage': memory_usage,
            'metrics': {
                'duration_seconds': duration,
                'packet_size_bytes': self.communication_size,
                'memory_usage_bytes': memory_usage
            }
        }
        self.logger.info(f"Finished communication for {self.operation_name}", extra=extra)
        # Reset for potential reuse
        self.communication_size = 0

def log_metrics(logger, metrics_dict, client_id=None, operation=None):
    """Log metrics in JSON format"""
    extra = {
        'client_id': client_id,
        'operation': operation,
        'event': 'metrics',
        'metrics': metrics_dict
    }
    logger.info("Metrics logged", extra=extra)

# --- Modified Logger Setup with JSON formatting ---
def setup_mqtt_logger(cid):
    logger = logging.getLogger(f"fl_client_{cid}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # MQTT Handler with JSON formatting
        mqtt_topic = f"{MQTT_BASE_TOPIC_CLIENT}/{cid}/logs"
        try:
            mqtt_handler = mqtthandler.MQTTHandler(
                host=MQTT_BROKER_HOST,
                topic=mqtt_topic,
                port=MQTT_BROKER_PORT,
                # qos=1,
                # retain=False,
                # client_id=f"fl_client_logger_{cid}"
            )
            
            # Use JSON formatter
            json_formatter = JSONFormatter()
            mqtt_handler.setFormatter(json_formatter)
            mqtt_handler.setLevel(logging.INFO)
            logger.addHandler(mqtt_handler)
            
            # Log initialization with client_id context
            extra = {'client_id': cid, 'event': 'logger_init'}
            logger.info(f"MQTT JSON logging initialized for client {cid} on topic {mqtt_topic}", extra=extra)
            
        except Exception as e:
            logger.error(f"Failed to initialize MQTT handler for client {cid}: {e}")
            # Fallback to console logging with JSON format
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                console_handler_fallback = logging.StreamHandler()
                console_handler_fallback.setFormatter(JSONFormatter())
                logger.addHandler(console_handler_fallback)
                extra = {'client_id': cid, 'event': 'fallback_logging'}
                logger.info("Fell back to console logging due to MQTT handler initialization error", extra=extra)
    
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

def train(net, trainloader, optimizer, epochs, device, logger=None, client_id=None):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    
    # Only log initial training start (no MQTT logging for intermediate operations)
    if logger:
        print(f"Starting training with {epochs} epochs for client {client_id}")
    
    for epoch_num in range(epochs):
        epoch_loss = 0.0
        
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
            
            epoch_loss += loss.item()
        
        print(f"Client {client_id}: Completed epoch {epoch_num + 1}/{epochs}, Loss: {epoch_loss / len(trainloader):.4f}")
    
    print(f"Client {client_id}: Training completed")


def test(net, testloader, device: str = "cpu", logger=None, client_id=None):
    """Validate the network on the testing set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    # Only log initial testing start (no MQTT logging for intermediate operations)
    if logger and client_id:
        print(f"Starting model evaluation for client {client_id}")
    
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
    avg_loss = loss / len(testloader)
    
    if client_id:
        print(f"Client {client_id}: Model evaluation completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy


def prepare_dataset(dataset, non_iid=False):
    """Get dataset and return client partitions and global testset."""
    print("Dataset: ", dataset)  # This print will not go to MQTT unless you change it
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
    
    for partition_id in range(NUM_CLIENTS):
        if dataset == "sc2":
            partition = partitions[partition_id]
        else:
            partition = fds.load_partition(partition_id, "train")
            
        train_val_split = partition.train_test_split(test_size=0.3, seed=42)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]

        train_data = train_data.with_transform(apply_transforms)
        val_data = val_data.with_transform(apply_transforms)
        
        trainsets.append(train_data)
        validsets.append(val_data)

    return trainsets, validsets

bitwidths = [8]

def post_training_quantization(model, test_dataloader, model_path):
    model.eval()
    quantizers = dict()
    for bitwidth in bitwidths:
        # model.load_state_dict(torch.load(model_path))
        print(f'k-means quantizing model into {bitwidth} bits')
        quantizer = KMeansQuantizer(model, bitwidth)
        quantized_model_size = model_size(model, bitwidth)
        print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size:.2f} bytes")
        _, quantized_model_accuracy = test(model, test_dataloader)
        print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy*100:.2f}%")
        quantizers[bitwidth] = quantizer
    return model, quantizers

def compare_model_sizes(original_model, quantized_model):
    original_size = model_size(original_model)
    print(f"\nOriginal model size: {original_size:.2f} bytes")
    
    original_params = model_params(original_model)
    quantized_params = model_params(quantized_model)
    
    for bitwidth in bitwidths:
        quantized_size = model_size(quantized_model, bitwidth)
        print(f"Bitwidth: {bitwidth} bits - Quantized model size: {quantized_size:.2f} bytes")
        print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")

    print(f"\nOriginal model parameters: {original_params:,}")
    print(f"Quantized model parameters: {quantized_params:,}")

def quantize_parameters_for_transmission(model, quantizer):
    """Convert model parameters to quantized format using cluster indices."""
    quantized_params = []
    cookbooks = []
    
    for name, param in model.named_parameters():
        # Get the codebook for this parameter
        codebook = quantizer.codebook[name]
        
        # Extract centroids and labels
        centroids = codebook.centroids.cpu().numpy()
        labels = codebook.labels.cpu().numpy()
        
        # Reshape labels to match parameter shape
        param_shape = param.shape
        indices = labels.reshape(param_shape).astype(np.uint8)
        
        quantized_params.append(indices)
        cookbooks.append(centroids)
    
    return quantized_params, cookbooks

def dequantize_parameters_from_indices(indices_list, cookbooks, model_state_dict):
    """Reconstruct parameters from quantized indices and cookbooks."""
    reconstructed_params = []
    
    for idx, (param_name, param_shape) in enumerate(model_state_dict.items()):
        indices = indices_list[idx]
        cookbook = cookbooks[idx]
        
        # Reconstruct parameter values from indices
        param_flat = indices.flatten()
        reconstructed_flat = np.array([cookbook[i] for i in param_flat])
        reconstructed = reconstructed_flat.reshape(param_shape.shape)
        
        reconstructed_params.append(reconstructed)
    
    return reconstructed_params

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, dataset, cid):
        self.cid = cid
        self.logger = setup_mqtt_logger(cid)
        
        EMBEDDING_DIM = 2
        HIDDEN_DIM = 100
        TAGSET_SIZE = 2
        
        self.trainset = trainset
        self.valset = valset
        
        if dataset in ["mnist", "fashion_mnist"]:
            self.model = LeNet5()
        elif dataset == "femnist":
            self.model = FEMNISTCNN()
        elif dataset == "cifar10":
            self.model = VeryDeepCNN()
        elif dataset == "sc2":
            self.model = BGRUTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)
        else:
            self.logger.warning(f"Dataset {dataset} not explicitly handled, using mobilenet_v3_small")
            self.model = mobilenet_v3_small(num_classes=10)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize quantizer immediately for all communication to be quantized
        self.quantizer = KMeansQuantizer(self.model, QUANTIZATION_BITS)
        
        # Log client initialization (kept as requested)
        extra = {
            'client_id': cid,
            'event': 'client_init',
            'metrics': {
                'dataset': dataset,
                'model_type': type(self.model).__name__,
                'device': str(self.device),
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'quantization_bits': QUANTIZATION_BITS
            }
        }
        self.logger.info(f"Client {cid} initialized with {QUANTIZATION_BITS}-bit quantization", extra=extra)

    def get_parameters(self, config):
        extra = {'client_id': self.cid, 'operation': 'get_parameters', 'event': 'operation_start'}
        self.logger.info("get_parameters called", extra=extra)
        
        # Always send quantized parameters
        indices_list, cookbooks = quantize_parameters_for_transmission(self.model, self.quantizer)
        
        # Calculate communication size (only indices count, not cookbooks)
        total_indices = sum(indices.size for indices in indices_list)
        param_bytes = total_indices  # 1 byte per index for 8-bit quantization
        
        with CommunicationTimer(self.logger, "get_parameters") as timer:
            timer.communication_size = param_bytes
            
            # Combine indices and cookbooks into parameters list
            # Format: [indices_0, cookbook_0, indices_1, cookbook_1, ...]
            parameters = []
            for indices, cookbook in zip(indices_list, cookbooks):
                parameters.append(indices)
                parameters.append(cookbook)
        
        extra = {
            'client_id': self.cid,
            'operation': 'get_parameters',
            'event': 'operation_complete',
            'metrics': {
                'parameter_bytes': param_bytes,
                'quantized': True,
                'total_indices': total_indices
            }
        }
        
        self.logger.info("get_parameters completed", extra=extra)
        return parameters

    def set_parameters(self, parameters):
        extra = {'client_id': self.cid, 'operation': 'set_parameters', 'event': 'operation_start'}
        self.logger.info("set_parameters called", extra=extra)
        
        # All parameters are quantized (alternating indices and cookbooks)
        # Extract indices and cookbooks
        indices_list = []
        cookbooks = []
        for i in range(0, len(parameters), 2):
            indices_list.append(parameters[i])
            cookbooks.append(parameters[i + 1])
        
        # Calculate communication size (only indices count)
        total_indices = sum(indices.size for indices in indices_list)
        param_bytes = total_indices  # 1 byte per index
        
        with CommunicationTimer(self.logger, "set_parameters") as timer:
            timer.communication_size = param_bytes
            
            # Dequantize parameters
            reconstructed_params = dequantize_parameters_from_indices(
                indices_list, cookbooks, self.model.state_dict()
            )
            
            # Set parameters
            params_dict = zip(self.model.state_dict().keys(), reconstructed_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            
            # Re-quantize the model with the new parameters to update the quantizer
            # This ensures the quantizer is consistent with the current model state
            self.quantizer = KMeansQuantizer(self.model, QUANTIZATION_BITS)
        
        extra = {
            'client_id': self.cid,
            'operation': 'set_parameters',
            'event': 'operation_complete',
            'metrics': {
                'parameter_bytes': param_bytes,
                'quantized': True,
                'total_indices': total_indices
            }
        }
        
        self.logger.info("set_parameters completed", extra=extra)

    def fit(self, parameters, config):
        # No MQTT logging for fit operation, only console output
        print(f"Client {self.cid}: Starting fit operation")
        
        self.set_parameters(parameters)
        
        batch_size, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device, 
              logger=None, client_id=self.cid)  # No MQTT logger passed
        
        # Re-quantize model after training to update cluster centers based on new weights
        print(f"Client {self.cid}: Re-quantizing model after training")
        self.quantizer = KMeansQuantizer(self.model, QUANTIZATION_BITS)
        
        print(f"Client {self.cid}: Fit operation completed")
        
        # Check unique values after quantization
        for name, param in self.model.named_parameters():
            codebook = self.quantizer.codebook[name]
            num_unique = len(codebook.centroids)
            print(f"{name}: {num_unique} unique values (clusters)")
        
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # No MQTT logging for evaluate operation, only console output
        print(f"Client {self.cid}: Starting evaluate operation")
        
        self.set_parameters(parameters)
        
        valloader = DataLoader(self.valset, batch_size=64, num_workers=0)
        loss, accuracy = test(self.model, valloader, device=self.device, 
                            logger=None, client_id=self.cid)  # No MQTT logger passed
        
        print(f"Client {self.cid}: Evaluate operation completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

def main():
    args = parser.parse_args()

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