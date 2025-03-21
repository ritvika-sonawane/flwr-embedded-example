import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from sc_quantized import BGRUTaggerQuantizable, prepare_scrimmage_data, train_brnn, val_brnn

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
    help="Dataset to use. Options: SC2, cifar10, mnist, fashion_mnist",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 10


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def prepare_dataset(dataset):
    """Get dataset and return client partitions and global testset."""
    print("Dataset: ", dataset)
    if dataset == "mnist" or dataset == "fashion_mnist":
        fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    elif dataset == "cifar10":
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == "sc2":
        partitions = prepare_scrimmage_data(NUM_CLIENTS)
    norm = Normalize((0.1307,), (0.3081,))
    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    for partition_id in range(NUM_CLIENTS):
        # partition = fds.load_partition(partition_id)
        partition = partitions[partition_id]
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=42)
        # partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    # testset = fds.load_split("test")
    # testset = testset.with_transform(apply_transforms)
    return trainsets, validsets, None


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset, dataset):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        if dataset in ["cifar10", "mnist", "fashion_mnist"]:
            self.model = Net()
        elif dataset == "sc2":
            self.model = BGRUTaggerQuantizable()
            # self.model = Net()
        else:
            self.model = mobilenet_v3_small(num_classes=10)
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # def fit(self, parameters, config):
    #     print("Client sampled for fit()")
    #     self.set_parameters(parameters)
    #     # Read hyperparameters from config set by the server
    #     batch, epochs = config["batch_size"], config["epochs"]
    #     # Construct dataloader
    #     trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=False)
    #     # Define optimizer
    #     optimizer = optim.Adam(self.model.parameters(), 0.0001)
    #     # Train
    #     # train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
    #     train_brnn(self.model, self.trainset, NUM_CLIENTS=3, NUM_EPOCHS=epochs, optimizer=optimizer)
    #     # Return local model and statistics
    #     return self.get_parameters({}), len(trainloader.dataset), {}

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]

        # Extract features (SNR, MCS) and labels (Frame_Error) as tensors
        train_df = self.trainset.to_pandas()
        train_x = torch.tensor(train_df[['SNR', 'MCS']].values, dtype=torch.float32)
        train_y = torch.tensor(train_df['Frame_Error'].values, dtype=torch.long)

        # Create DataLoader from the partition
        train_dataset = data.TensorDataset(train_x, train_y)
        trainloader = data.DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True)

        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), 0.0001)

        # Train using the DataLoader
        train_brnn(self.model, NUM_EPOCHS=epochs, trainloader=trainloader, optimizer=optimizer)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    # def evaluate(self, parameters, config):
    #     print("Client sampled for evaluate()")
    #     self.set_parameters(parameters)
    #     # Construct dataloader
    #     # Extract features (SNR, MCS) and labels (Frame_Error) as tensors
    #     val_x = torch.tensor(self.valset[['SNR', 'MCS']].to_pandas().values, dtype=torch.float32)
    #     val_y = torch.tensor(self.valset['Frame_Error'].to_pandas().values, dtype=torch.long)

    #     # Create DataLoader from the partition
    #     import torch.utils.data as data
    #     # valloader = DataLoader(self.valset, batch_size=64)
    #     val_dataset = data.TensorDataset(val_x, val_y)
    #     valloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    #     # Evaluate
    #     loss, accuracy = test(self.model, valloader, device=self.device)
    #     # Return statistics
    #     return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        val_df = self.valset.to_pandas()
        val_x = torch.tensor(val_df[['SNR', 'MCS']].values, dtype=torch.float32)
        val_y = torch.tensor(val_df['Frame_Error'].values, dtype=torch.long)
        valloader = data.DataLoader(data.TensorDataset(val_x, val_y), batch_size=64)
        
        loss, accuracy = test(self.model, valloader, self.device)
        val_brnn(self.model, valloader=valloader)
        return float(loss), len(valloader.dataset), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    dataset = args.dataset.lower()
    # Download dataset and partition itn
    trainsets, valsets, _ = prepare_dataset(dataset)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], dataset=dataset
        ).to_client(),
    )


if __name__ == "__main__":
    main()
