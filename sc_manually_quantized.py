import torch
import torch.nn as nn
import torch.quantization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
import os
from models import MLP, BGRUTagger, model_size, model_params
from quantize.k_means import KMeansQuantizer

bitwidths = [8, 4, 2, 1]

def get_data_data():
    with open('sc2/scrimmage4_link_dataset.pickle', 'rb') as file:
        link_dataset = pickle.load(file)
    return len(link_dataset), link_dataset

def prepare_scrimmage_data(NUM_CLIENTS):
    _, link_dataset = get_data_data()

    selected_cols = [0, 1]  # Only SNR and MCS
    features = np.vstack([data[0][:, selected_cols] for data in link_dataset])
    labels = np.hstack([data[1] for data in link_dataset])

    df = pd.DataFrame(features, columns=['SNR', 'MCS'])
    df['Frame_Error'] = labels

    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    partitioner = IidPartitioner(num_partitions=NUM_CLIENTS)
    partitioner.dataset = dataset

    partitions = [partitioner.load_partition(i) for i in range(NUM_CLIENTS)]
    return partitions

def train(model, trainloader, val_dataloader, NUM_EPOCHS, optimizer):
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    for epoch_idx in range(NUM_EPOCHS): 
        progress_training_epoch = tqdm(
            trainloader, 
            desc=f'Epoch {epoch_idx+1}/{NUM_EPOCHS}, Training',
            miniters=1, ncols=88, position=0,
            leave=True, total=len(trainloader), smoothing=.9)
        train_loss = 0
        train_size = 0
        model.train()
        for idx, item in enumerate(progress_training_epoch):
            sentence, tags = item
            model.zero_grad()
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()
            train_loss += loss * tags.size()[0]
            train_size += tags.size()[0]
        print(f'Train loss:{train_loss.item()/train_size: .5f})')
        validate(model, val_dataloader)


def validate(model, valloader):
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    for epoch_idx in range(1): 
        test_loss = 0
        test_size = 0
        predict = []
        target = []
        progress_validation_epoch = tqdm(
            valloader, 
            desc=f'Epoch {epoch_idx+1}/{1}, Validation',
            miniters=1, ncols=88, position=0, 
            leave=True, total=len(valloader), smoothing=.9)
        model.eval()
        with torch.no_grad():
            for idx, (sentence, tags) in enumerate(progress_validation_epoch):
                tag_scores = model(sentence)
                loss = loss_function(tag_scores, tags)
                predict.append(tag_scores.argmax(dim=1).numpy())
                target.append(tags.numpy())        
                test_loss += loss * tags.size()[0]
                test_size += tags.size()[0]
        predict = np.concatenate(predict, axis=0)
        target = np.concatenate(target, axis=0)

        print(f'Validation loss:{test_loss.item()/test_size: .5f}')

def post_training_quantization(model, test_dataloader):
    model.eval()
    quantizers = dict()
    for bitwidth in bitwidths:
        model.load_state_dict(torch.load("models/original_model.pt", map_location=torch.device('cpu')))
        print(f'k-means quantizing model into {bitwidth} bits')
        quantizer = KMeansQuantizer(model, bitwidth)
        quantized_model_size = model_size(model, bitwidth)
        print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size:.2f} bytes")
        quantized_model_accuracy = test(model, test_dataloader)
        print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}%")
        # print(quantizer.codebook)
        quantizers[bitwidth] = quantizer
    return model

def test(model, test_dataloader):
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    
    test_loss = 0
    test_size = 0
    predict = []
    target = []
    
    model.eval()
    for name, param in model.named_parameters():
        unique_values = torch.unique(param.data)
        print(f"Parameter: {name}")
        print(f"Unique values ({len(unique_values)}): {unique_values.cpu().numpy()}\n")

    with torch.no_grad():
        for sentence, tags in tqdm(test_dataloader, desc="Testing"):
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, tags)
            predict.append(tag_scores.argmax(dim=1).numpy())
            target.append(tags.numpy())
            test_loss += loss * tags.size()[0]
            test_size += tags.size()[0]
    
    predict = np.concatenate(predict, axis=0)
    target = np.concatenate(target, axis=0)
    
    accuracy = (predict == target).mean()
    
    print(f'\nModel test loss: {test_loss.item()/test_size:.5f}')
    print(f'Model accuracy: {accuracy:.5f}')
    
    return accuracy*100

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

def main():
    EMBEDDING_DIM = 2
    HIDDEN_DIM = 100
    TAGSET_SIZE = 2
    NUM_EPOCHS = 1
    TRAIN_BATCH_SIZE = 1024
    VAL_BATCH_SIZE = 128

    partitions = prepare_scrimmage_data(NUM_CLIENTS=2)
    
    partition_df = partitions[0].to_pandas()
    
    train_df, temp_df = train_test_split(partition_df, test_size=0.6, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.83, shuffle=False)
    
    train_x = torch.tensor(train_df[['SNR', 'MCS']].values, dtype=torch.float32)
    train_y = torch.tensor(train_df['Frame_Error'].values, dtype=torch.long)
    
    val_x = torch.tensor(val_df[['SNR', 'MCS']].values, dtype=torch.float32)
    val_y = torch.tensor(val_df['Frame_Error'].values, dtype=torch.long)
    
    test_x = torch.tensor(test_df[['SNR', 'MCS']].values, dtype=torch.float32)
    test_y = torch.tensor(test_df['Frame_Error'].values, dtype=torch.long)
    
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_x, val_y),
        batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=True
    )
    
    model = BGRUTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)
    # model = MLP()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train(model, train_dataloader, val_dataloader, NUM_EPOCHS, optimizer)
    validate(model, val_dataloader)
    test(model, test_dataloader)

    torch.save(model.state_dict(), "models/original_model.pt")
        
    model.load_state_dict(
        torch.load("models/original_model.pt", map_location=torch.device('cpu'))
    )
    
    quantized_model = post_training_quantization(model, test_dataloader)

    torch.save(quantized_model.state_dict(), "models/quantized_model.pt") 
    
    accuracy = test(quantized_model, test_dataloader)
    print(f"Quantized model accuracy: {accuracy:.2f}")
    
    compare_model_sizes(model, quantized_model)
    
if __name__ == "__main__":
    main()