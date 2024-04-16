import torch
import torch.nn as nn
from torchvision import models

class Deep_opt(nn.Module):
    def __init__(self, hidden_size):
        super(Deep_opt, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs):
        attention_scores = torch.softmax(self.fc(rnn_outputs).squeeze(-1), dim=1)
        weighted_features = torch.sum(rnn_outputs * attention_scores.unsqueeze(-1), dim=1)
        return weighted_features

class DeepHybridnet(nn.Module):
    def __init__(self, num_classes, hidden_size=256, lstm_layers=2):
        super(DeepHybridnet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2] 
        self.resnet_extractor = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.attention = Deep_opt(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch_size, seq_length, C, H, W = x.size()
        cnn_features = torch.zeros(batch_size, seq_length, 2048).to(x.device)
        for t in range(seq_length):
            with torch.no_grad():
                feature = self.resnet_extractor(x[:, t, :, :, :])
                feature = self.avgpool(feature)
                feature = feature.view(feature.size(0), -1)
                cnn_features[:, t, :] = feature
        lstm_out, _ = self.lstm(cnn_features)
        attention_out = self.attention(lstm_out)
        out = self.fc(attention_out)
        return out

