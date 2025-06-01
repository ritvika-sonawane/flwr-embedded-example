# python server.py --rounds 3 --num_clients 5 --local_ep 1 --min_num_clients 2 --sample_fraction 0.2 --server_address="0.0.0.0:8080"

import argparse
import logging
from typing import List, Tuple
import mqtthandler
import paho.mqtt.client as mqtt

import flwr as fl
from flwr.common import Metrics

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_SERVER = "federated_learning/server/logs"

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
    default=10,
    help="Total Number of clients for sampling (default: 5)",
)

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

def main():
    args = parser.parse_args()
    server_logger.info("Starting Flower server with arguments: %s", args)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit = args.sample_fraction,
        fraction_evaluate = args.sample_fraction,
        min_fit_clients = args.min_num_clients,
        min_available_clients = args.num_clients,
        on_fit_config_fn = generate_fit_config,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    server_logger.info("Flower server strategy configured. Starting server...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    server_logger.info("Flower server stopped.")


if __name__ == "__main__":
    main()