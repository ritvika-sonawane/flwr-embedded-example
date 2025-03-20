import torch
import torch.nn as nn
import os
from sru import SRU, SRUCell

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 50)
        self.selu = nn.SELU()
        self.linear2 = nn.Linear(50, 50)
        self.selu = nn.SELU()
        self.linear3 = nn.Linear(50, 50)
        self.selu = nn.SELU()        
        self.drop1 = nn.AlphaDropout(p=0.2)
        self.linear5 = nn.Linear(50,2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        output = self.linear1(input)
        output = self.selu(output)
        output = self.drop1(output)
        output = self.linear2(output)
        output = self.selu(output)
        output = self.drop1(output)
        output = self.linear3(output)
        output = self.selu(output)
        output = self.drop1(output)
        output = self.linear5(output)
        output = self.softmax(output)
        return output

class BGRUTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(BGRUTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.bgru = nn.GRU(embedding_dim, hidden_dim,bidirectional=True)        
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
            
    def forward(self, sentence):
        gru_out, _ = self.bgru(sentence.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(gru_out.view(len(sentence), -1))
        return tag_space

class CBSDNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.conv1.weight)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.conv2.weight)

        self.linear1 = nn.Linear(16*2, 100)
        
        self.bsru = SRU(input_size=100, hidden_size=100,num_layers=2,bidirectional=True)
        self.linear2 = nn.Linear(100*2, 100*2)
        nn.init.xavier_normal_(self.linear2.weight)

        self.linear3 = nn.Linear(100*2, 2)

        self.relu = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        
        output = self.conv1(input)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.dropout1(output)
                
        output = output.view(-1, 16*2)
        output1 = self.linear1(output)

        output = output1.view(len(output1),1 ,-1 )
        output, _ = self.bsru(output)
        
        output = output.view(len(output1), -1)

        output = self.linear2(output)
        output = self.relu(output)

        output = self.dropout2(output)

        output = self.linear3(output)
        return output

def model_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size(model, bitwidth=32):
    return model_params(model) * bitwidth