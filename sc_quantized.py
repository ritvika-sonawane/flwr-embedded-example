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
from models import MLP

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

def train_brnn(model, trainloader, NUM_EPOCHS, optimizer):
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
        print(f'train loss:{train_loss.item()/train_size: .5f})')

def val_brnn(model, valloader):
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

        print(f'validation loss:{test_loss.item()/test_size: .5f}')

def apply_post_training_quantization(model, calibration_dataloader):
    model.eval()
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    
    print("Calibrating model with data...")
    with torch.no_grad():
        for sentences, _ in tqdm(calibration_dataloader):
            model(sentences)
    
    torch.quantization.convert(model, inplace=True)
    return model

def test_quantized_model(quantized_model, test_dataloader):
    """Test the quantized model and compare with original model"""
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    
    test_loss = 0
    test_size = 0
    predict = []
    target = []
    
    quantized_model.eval()
    with torch.no_grad():
        for sentence, tags in tqdm(test_dataloader, desc="Testing quantized model"):
            tag_scores = quantized_model(sentence)
            loss = loss_function(tag_scores, tags)
            predict.append(tag_scores.argmax(dim=1).numpy())
            target.append(tags.numpy())
            test_loss += loss * tags.size()[0]
            test_size += tags.size()[0]
    
    predict = np.concatenate(predict, axis=0)
    target = np.concatenate(target, axis=0)
    
    accuracy = (predict == target).mean()
    
    tp = predict[target==1].sum()
    tn = (predict[target==0] == 0).sum()
    fp = predict[target==0].sum()
    fn = [target==1].sum() - predict[target==1].sum()
    
    print(f'Quantized model test loss: {test_loss.item()/test_size:.5f}')
    print(f'Quantized model accuracy: {accuracy:.5f}')
    print(f'True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}')
    
    return accuracy

def compare_model_sizes(original_model, quantized_model):
    torch.save(original_model.state_dict(), "original_model.pt")
    original_size = os.path.getsize("original_model.pt") / (1024 * 1024)
    
    torch.save(quantized_model.state_dict(), "quantized_model.pt") 
    quantized_size = os.path.getsize("quantized_model.pt") / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")
    
    return original_size, quantized_size

def main():
    EMBEDDING_DIM = 2
    HIDDEN_DIM = 100
    TAGSET_SIZE = 2
    NUM_EPOCHS = 1
    BATCH_SIZE = 128

    partitions = prepare_scrimmage_data(NUM_CLIENTS=10)
    
    partition_df = partitions[0].to_pandas()
    
    train_df, temp_df = train_test_split(partition_df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_x = torch.tensor(train_df[['SNR', 'MCS']].values, dtype=torch.float32)
    train_y = torch.tensor(train_df['Frame_Error'].values, dtype=torch.long)
    
    val_x = torch.tensor(val_df[['SNR', 'MCS']].values, dtype=torch.float32)
    val_y = torch.tensor(val_df['Frame_Error'].values, dtype=torch.long)
    
    test_x = torch.tensor(test_df[['SNR', 'MCS']].values, dtype=torch.float32)
    test_y = torch.tensor(test_df['Frame_Error'].values, dtype=torch.long)
    
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_x, val_y),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # model = BGRUTaggerQuantizable(EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE)
    model = MLP()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_brnn(model, train_dataloader, NUM_EPOCHS, optimizer)
    val_brnn(model, val_dataloader)
    torch.save(model.state_dict(), "original_model.pt")
        
    # Load weights
    # model.load_state_dict(
    #     torch.load("original_model.pt", map_location=torch.device('cpu'))
    # )
    original_model = model
    
    quantized_model = apply_post_training_quantization(model, val_dataloader)
    
    accuracy = test_quantized_model(quantized_model, test_dataloader)
    print(f"Quantized model accuracy: {accuracy:.2f}")
    
    compare_model_sizes(original_model, quantized_model)
    
    torch.save(quantized_model.state_dict(), "bgru_quantized.pt")
    print("Quantized model saved as 'bgru_quantized.pt'")

if __name__ == "__main__":
    main()