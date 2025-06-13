# python server.py --rounds 3 --num_clients 5 --local_ep 1 --min_num_clients 2 --sample_fraction 0.2 --server_address="0.0.0.0:8080"

import argparse
import logging
from typing import List, Tuple
import mqtthandler
import paho.mqtt.client as mqtt
import numpy as np
from collections import OrderedDict
import torch

import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from quantize.k_means import k_means_quantize, Codebook

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_SERVER = "federated_learning/server/logs"
QUANTIZATION_BITS = 8

# --- Configure Logging ---
def setup_server_mqtt_logging():
    logger = logging.getLogger("fl_server") # server logger
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console Handler (optional)
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        # logger.addHandler(console_handler)

        # MQTT Handler
        try:
            mqtt_handler = mqtthandler.MQTTHandler(
                host=MQTT_BROKER_HOST,
                topic=MQTT_TOPIC_SERVER,
                port=MQTT_BROKER_PORT,
                # qos=1,
                # client_id="fl_server_logger"
            )
            mqtt_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            mqtt_handler.setLevel(logging.INFO)
            logger.addHandler(mqtt_handler)
            logger.info(f"MQTT logging for server initialized on topic {MQTT_TOPIC_SERVER}")
        except Exception as e:
            logger.error(f"Failed to initialize MQTT handler for server: {e}")
            # Fallback to basic console logging for the server logger
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                console_handler_fallback = logging.StreamHandler()
                console_handler_fallback.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - Fallback - %(message)s"))
                logger.addHandler(console_handler_fallback)
                logger.info("Server fell back to console logging due to MQTT handler initialization error.")
    return logger

# Get the server logger instance
server_logger = setup_server_mqtt_logging()

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--local_ep",
    type=int,
    default=1,
    help="Number of local epochs of federated learning (default: 1)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=0.2,
    help="Fraction of available clients used for fit/evaluate (default: 0.2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--num_clients",
    type=int,
    default=5,
    help="Total Number of clients for sampling (default: 5)",
)

def dequantize_parameters(quantized_params_list):
    """Dequantize parameters from indices and cookbooks format to full precision values."""
    # Extract indices and cookbooks
    num_layers = len(quantized_params_list) // 2
    dequantized_params = []
    
    for layer_idx in range(num_layers):
        indices_idx = layer_idx * 2
        cookbook_idx = layer_idx * 2 + 1
        
        indices = quantized_params_list[indices_idx]
        cookbook = quantized_params_list[cookbook_idx]
        
        # Reconstruct full precision values from indices and cookbook
        param_flat = indices.flatten()
        reconstructed_flat = np.array([cookbook[i] for i in param_flat])
        reconstructed = reconstructed_flat.reshape(indices.shape)
        
        dequantized_params.append(reconstructed)
    
    return dequantized_params

def quantize_parameters(params_list):
    """Quantize full precision parameters to indices and cookbooks format."""
    quantized_params = []
    
    for param in params_list:
        # Convert to tensor for k-means quantization
        param_tensor = torch.tensor(param, dtype=torch.float32)
        
        # Apply k-means quantization
        codebook = k_means_quantize(param_tensor.clone(), bitwidth=QUANTIZATION_BITS)
        
        # Extract indices and centroids
        indices = codebook.labels.cpu().numpy().reshape(param.shape).astype(np.uint8)
        centroids = codebook.centroids.cpu().numpy()
        
        # Add to quantized params (indices, then cookbook)
        quantized_params.append(indices)
        quantized_params.append(centroids)
    
    return quantized_params

def aggregate_parameters(results):
    """Aggregate parameters using standard FedAvg after dequantization."""
    if not results:
        return None
    
    # Dequantize all client parameters
    weights_results = []
    for _, fit_res in results:
        quantized_params = parameters_to_ndarrays(fit_res.parameters)
        dequantized_params = dequantize_parameters(quantized_params)
        weights_results.append((dequantized_params, fit_res.num_examples))
    
    # Standard FedAvg aggregation on full precision parameters
    num_examples_total = sum([num_examples for _, num_examples in weights_results])
    
    # Initialize aggregated parameters
    aggregated_params = []
    num_layers = len(weights_results[0][0])
    
    for layer_idx in range(num_layers):
        # Weighted average for each layer
        layer_aggregated = None
        
        for params, num_examples in weights_results:
            weight = num_examples / num_examples_total
            layer_param = params[layer_idx]
            
            if layer_aggregated is None:
                layer_aggregated = weight * layer_param
            else:
                layer_aggregated += weight * layer_param
        
        aggregated_params.append(layer_aggregated)
    
    # Quantize aggregated parameters before sending back
    quantized_aggregated = quantize_parameters(aggregated_params)
    
    return ndarrays_to_parameters(quantized_aggregated)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Log details of metrics received
    for i, (num_ex, m) in enumerate(metrics):
        server_logger.info(f"Metrics from client {i} (or unidentified client): {num_ex} examples, accuracy {m.get('accuracy', 'N/A')}")
        
    aggregated_accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0
    server_logger.info(f"Aggregated weighted average accuracy: {aggregated_accuracy:.4f}")
    return {"accuracy": aggregated_accuracy}

def generate_fit_config(server_round: int):
    args = parser.parse_args([]) # empty list if not expecting args
    config = {
        "server_round": server_round,
        "num_clients": args.num_clients,
        "epochs": args.local_ep,
        "batch_size": 16,
    }
    server_logger.info(f"Dispatching fit config for round {server_round}: {config}")
    return config

class QuantizedFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that handles quantized communication."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate fit results with proper dequantization and requantization."""
        if not results:
            return None, {}
        
        # Log aggregation info
        server_logger.info(f"Aggregating {len(results)} fit results for round {server_round}")
        
        # Aggregate parameters with dequantization and requantization
        aggregated_parameters = aggregate_parameters(results)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        
        return aggregated_parameters, metrics_aggregated

def main():
    args = parser.parse_args()
    server_logger.info("Starting Flower server with arguments: %s", args)

    strategy = QuantizedFedAvg(
        fraction_fit = args.sample_fraction,
        fraction_evaluate = args.sample_fraction,
        min_fit_clients = args.min_num_clients,
        min_available_clients = args.num_clients,
        on_fit_config_fn = generate_fit_config,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    server_logger.info("Flower server strategy configured for quantized communication. Starting server...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    server_logger.info("Flower server stopped.")


if __name__ == "__main__":
    main()