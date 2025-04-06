import torch
import torch.nn as nn
import torch.nn.functional as F

# from sru import SRU, SRUCell

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self, in_channels=1) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
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

# class CBSDNN(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         nn.init.xavier_normal_(self.conv1.weight)

#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
#         nn.init.xavier_normal_(self.conv2.weight)

#         self.linear1 = nn.Linear(16*2, 100)
        
#         self.bsru = SRU(input_size=100, hidden_size=100,num_layers=2,bidirectional=True)
#         self.linear2 = nn.Linear(100*2, 100*2)
#         nn.init.xavier_normal_(self.linear2.weight)

#         self.linear3 = nn.Linear(100*2, 2)

#         self.relu = nn.ReLU()

#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.2)

#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input):
        
#         output = self.conv1(input)
#         output = self.relu(output)

#         output = self.conv2(output)
#         output = self.relu(output)
#         output = self.dropout1(output)
                
#         output = output.view(-1, 16*2)
#         output1 = self.linear1(output)

#         output = output1.view(len(output1),1 ,-1 )
#         output, _ = self.bsru(output)
        
#         output = output.view(len(output1), -1)

#         output = self.linear2(output)
#         output = self.relu(output)

#         output = self.dropout2(output)

#         output = self.linear3(output)
#         return output

def model_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size(model, bitwidth=32):
    return model_params(model) * bitwidth

# model_dict[0]: [[32, 3, 1], [64, 3, 1], "M",'F','D', 128]
class SimpleCNN(nn.Module):
    """Model 0: Simple CNN with 2 conv layers"""
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return self.fc2(x)

# model_dict[8]: [[64,3, 1], [64, 3, 1], 'M', [128,3,1], [128,3,1], [128,3,1], 'M', 'F', 'D', 128]
class DeepCNN(nn.Module):
    """Model 8: Deeper CNN with 5 conv layers"""
    def __init__(self, in_channels=3, num_classes=10):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return self.fc2(x)

# model_dict[1]: [[128,3, 1], [128, 3, 1], 'M', [256,3,1], [256,3,1], 'M', [512,3,1], [512, 3,1], 'M', 'F', 'D', 128]
class VeryDeepCNN(nn.Module):
    """Model 1: Very deep CNN with 4 conv layers"""
    def __init__(self, in_channels=3, num_classes=10):
        super(VeryDeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(512 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return self.fc2(x)

# model_dict[4]: [[6, 5, 2], "M", [16, 5, 0], "M", 'F', 120, 84]
class LeNet5(nn.Module):
    """Model 4: LeNet-5 architecture"""
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# model_dict[7]: [[32, 5, 2],"M", [64, 5, 2], "M", 'F', 2048]
class FEMNISTCNN(nn.Module):
    """Model 7: CNN for FEMNIST dataset"""
    def __init__(self, in_channels=1, num_classes=62):
        super(FEMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Calculate the size of flattened features
        # After conv1: 28x28 (MNIST/FEMNIST size)
        # After pool1: 14x14
        # After conv2: 14x14
        # After pool2: 7x7
        # Final size: 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Return only the output tensor, not a tuple

# model_dict[5]: [24,24]
class SimpleMLP(nn.Module):
    """Model 5: Simple 2-layer network"""
    def __init__(self, input_size=784, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# model_dict[6]: [32, 16, 32]
class ThreeLayerMLP(nn.Module):
    """Model 6: Simple 3-layer network"""
    def __init__(self, input_size=784, num_classes=10):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)