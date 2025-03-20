import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner

class BGRUTagger(nn.Module):

    def __init__(self, EMBEDDING_DIM = 2, HIDDEN_DIM = 100, TAGSET_SIZE = 2):
        super(BGRUTagger, self).__init__()
        
        self.hidden_dim = HIDDEN_DIM
        self.bgru = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM,bidirectional=True)        
        self.hidden2tag = nn.Linear(HIDDEN_DIM*2, TAGSET_SIZE)
            
    def forward(self, sentence):
        gru_out, _ = self.bgru(sentence.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(gru_out.view(len(sentence), -1))
        return tag_space
    
def get_data_data():
    with open('sc2/scrimmage4_link_dataset.pickle', 'rb') as file:
        link_dataset = pickle.load(file)
    return len(link_dataset), link_dataset

def prepare_scrimmage_data(NUM_CLIENTS):

    """Loads Scrimmage 4 dataset and partitions it using Flower's Partitioner."""
    
    # Load dataset from pickle
    _, link_dataset = get_data_data()

    # Convert dataset into NumPy arrays
    selected_cols = [0, 1]  # Only SNR and MCS
    features = np.vstack([data[0][:, selected_cols] for data in link_dataset])
    labels = np.hstack([data[1] for data in link_dataset])

    # Convert to Pandas DataFrame
    df = pd.DataFrame(features, columns=['SNR', 'MCS'])
    df['Frame_Error'] = labels

    # Convert to Hugging Face Dataset for Flower partitioning
    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    # Apply Flower's ChosenPartitioner
    partitioner = IidPartitioner(num_partitions=NUM_CLIENTS)
    partitioner.dataset = dataset

    partitions = [partitioner.load_partition(i) for i in range(NUM_CLIENTS)]
    return partitions


    # ########## Numpy DataLoader ##########
    # with open('scrimmage4_link_dataset.pickle', 'rb') as file:
    #     link_dataset = pickle.load(file)
    # total_links = len(link_dataset)
    # # Convert dataset into NumPy arrays
    # # Columns : 0 - SNR, 1 - MCS, 2 - Center Frequency, 3 - Bandwidth, [4:19] - PSD
    # selected_cols = [0, 1] 
    # features = np.vstack([data[0][:, selected_cols] for data in link_dataset])
    # labels = np.hstack([data[1] for data in link_dataset])

    # df = pd.DataFrame(features, columns=['SNR', 'MCS'])
    # df['Frame_Error'] = labels

    # train_val_size = int(0.5 * len(df))  # 50% 
    # train_size = int(0.4 * len(df))  # Train 40%
    # val_size = train_val_size - train_size  # Validation 10%

    # train_data, test_data = train_test_split(df, test_size=0.5, random_state=42, stratify=df['Frame_Error'])
    # train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42, stratify=train_data['Frame_Error'])

    # train_x, train_y = train_data.drop(columns=['Frame_Error']).values, train_data['Frame_Error'].values
    # val_x, val_y = val_data.drop(columns=['Frame_Error']).values, val_data['Frame_Error'].values
    # test_x, test_y = test_data.drop(columns=['Frame_Error']).values, test_data['Frame_Error'].values

    ########## Tensor DataLoader ##########

    # # Retain only the features selected by the RFE method
    # # Columns : 0 - SNR, 1 - MCS, 2 - Center Frequency, 3 - Bandwidth, [4:19] - PSD
    # cols = torch.LongTensor([0,1]) 
    # link_data = [(link_datas[0][:,cols], link_datas[1]) for link_datas in link_dataset]

    # # Without Pilot
    # test_start = 142
    # total_links = len(link_dataset)
    # # Do a 40:10:50 train:validation:test split 
    # train_data = link_data[:114]
    # val_data = link_data[114:142]
    # test_data = link_data[142:]

    # train_x = torch.cat(tuple(link[0] for link in train_data),dim=0)
    # train_y = torch.cat(tuple(link[1] for link in train_data),dim=0)

    # val_x = torch.cat(tuple(link[0] for link in val_data),dim=0)
    # val_y = torch.cat(tuple(link[1] for link in val_data),dim=0)

    # test_x = torch.cat(tuple(link[0] for link in test_data),dim=0)
    # test_y = torch.cat(tuple(link[1] for link in test_data),dim=0)

    # return total_links, link_dataset, train_x, train_y, val_x, val_y, test_x, test_y

def train_brnn(model, trainloader, valloader, NUM_EPOCHS, optimizer):
    
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]))
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
            # print(item)
            sentence, tags = item
            # sentence = sentence.cuda()
            # tags = tags.cuda()
            model.zero_grad()
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()
            train_loss += loss * tags.size()[0]
            train_size += tags.size()[0]
        
        test_loss = 0
        test_size = 0
        predict = []
        target = []
        progress_validation_epoch = tqdm(
            valloader, 
            desc=f'Epoch {epoch_idx+1}/{NUM_EPOCHS}, Validation',
            miniters=1, ncols=88, position=0, 
            leave=True, total=len(valloader), smoothing=.9)
        model.eval()
        with torch.no_grad():
            for idx, (sentence, tags) in enumerate(progress_validation_epoch):
                # sentence = sentence.cuda()
                # tags = tags.cuda()
                tag_scores = model(sentence)
                loss = loss_function(tag_scores, tags)
                predict.append(tag_scores.argmax(dim=1).numpy())
                target.append(tags.numpy())        
                test_loss += loss * tags.size()[0]
                test_size += tags.size()[0]
        predict = np.concatenate(predict, axis=0)
        target = np.concatenate(target, axis=0)

        print(f'train loss:{train_loss.item()/train_size: .5f}, '
            f'validation loss:{test_loss.item()/test_size: .5f}')

def test_brnn(model, NUM_EPOCHS):
    total_links, link_data = get_data_data()
    _, _, test_data = prepare_scrimmage_data(0)
    test_start = 142
    progress_test_epoch = tqdm(
    test_dataloader, 
    desc=f'Epoch {1}/{NUM_EPOCHS}, Test',
    miniters=1, ncols=88, position=0, 
    leave=True, total=len(test_dataloader), smoothing=.9)

    predict = []
    target = []
    snrss = []
    model.eval()
    with torch.no_grad():
        for idx, (sentence, tags) in enumerate(progress_test_epoch):
            # sentence = sentence.cuda()
            # tags = tags.cuda()
            tag_scores = model(sentence)
            predict.append(tag_scores.argmax(dim=1).numpy())
            target.append(tags.numpy())
            snrss.append(sentence.numpy()[:,0])
            
    predict = np.concatenate(predict, axis=0)
    target = np.concatenate(target, axis=0)
    snrss = np.concatenate(snrss, axis=0)

    accr = []

    for i in range(total_links - test_start):
        test_data_i = [link_data[i+test_start]]
        test_x_i = torch.cat(tuple(link[0] for link in test_data_i),dim=0)
        test_y_i = torch.cat(tuple(link[1] for link in test_data_i),dim=0)
        
        test_dataloader = data.DataLoader(
        data.TensorDataset(test_x_i,test_y_i), 
        batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
        
        progress_test_epoch = tqdm(
            test_data_i, 
            desc=f'Link {i+test_start+1}/{total_links}, Test',
            miniters=1, ncols=88, position=0, 
            leave=True, total=len(test_data_i), smoothing=.9)

        predict = []
        target = []
        model.eval()
        with torch.no_grad():
            for idx, (sentence, tags) in enumerate(progress_test_epoch):
                # sentence = sentence.cuda()
                # tags = tags.cuda()
                tag_scores = model(sentence)
                predict.append(tag_scores.argmax(dim=1).numpy())
                target.append(tags.numpy())

        predict = np.concatenate(predict, axis=0)
        target = np.concatenate(target, axis=0)

        tp = predict[target==1].sum()
        tn = (target==0).sum() - predict[target==0].sum()
        fp = predict[target==0].sum()
        fn = (target==1).sum() - predict[target==1].sum()
        accr.append((tp+tn)/(tp+tn+fp+fn))

        return 0, np.mean(accr)