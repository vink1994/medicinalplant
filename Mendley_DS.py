import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensors
])
train_dataset = ImageFolder(root='G:/vineet/medicinal_plant/Code/V1/mendley_dataset/split_dataset/train', transform=transform)
test_dataset = ImageFolder(root='G:/vineet/medicinal_plant/Code/V1/mendley_dataset/split_dataset/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

class Deep_opt(nn.Module):

    def __init__(self, hidden_size):
        super(Deep_opt, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_length, hidden_size)
        attention_scores = torch.softmax(self.fc(rnn_outputs).squeeze(-1), dim=1)
        # (batch_size, seq_length, 1)
        weighted_features = torch.sum(rnn_outputs * attention_scores.unsqueeze(-1), dim=1)
        # (batch_size, hidden_size)
        return weighted_features

class DeepHybridnet(nn.Module):
    def __init__(self, num_classes, hidden_size=256, lstm_layers=2):
        super(DeepHybridnet, self).__init__()
        # Use ResNet50 pretrained on ImageNet as the feature extractor
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove the last fully connected layer and pooling layer
        self.resnet_extractor = nn.Sequential(*modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Pool the features down to 1x1
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.attention = Deep_opt(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 4:  # Add a sequence dimension if missing
            x = x.unsqueeze(1)

        batch_size, seq_length, C, H, W = x.size()
        cnn_features = torch.zeros(batch_size, seq_length, 2048).to(x.device)

        for t in range(seq_length):
            with torch.no_grad():  # No need to track gradients for the pre-trained CNN part
                feature = self.resnet_extractor(x[:, t, :, :, :])
                feature = self.avgpool(feature)
                feature = feature.view(feature.size(0), -1)
                cnn_features[:, t, :] = feature

        lstm_out, _ = self.lstm(cnn_features)
        attention_out = self.attention(lstm_out)
        out = self.fc(attention_out)

        return out

# Assuming the CNNRNNModel class definition is already provided above
if __name__ == '__main__':
    # Your code for training or evaluation here

# Initialize the device to use for model training (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the selected device
    num_classes = 30  # Ensure this matches the number of classes in your dataset
    model = DeepHybridnet(num_classes=num_classes).to(device)

# Define the loss function
    criterion = nn.CrossEntropyLoss()

# Define the optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepHybridnet(num_classes=30).to(device)  # Adjust num_classes based on your dataset
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_classes = len(train_dataset.classes)  # Assuming train_dataset is your training dataset
    print(f"Number of classes: {num_classes}")

# Make sure your model's last layer matches this number
    model = DeepHybridnet(num_classes=num_classes).to(device)
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    print(f"Min label: {min(all_labels)}, Max label: {max(all_labels)}")
 # Assuming the DeepHybridnet class is defined and available

# Initialize the model structure
    model = DeepHybridnet(num_classes=30).to(device)

# Load the saved model parameters
    model.load_state_dict(torch.load('G:/vineet/medicinal_plant/Code/V1/weights/mode.pth', map_location=torch.device('cpu')))


# Switch to evaluation mode
    model.eval()

# Proceed with evaluation using test_loader as shown in the Evaluation cell
    correct = 0
    total = 0

    with torch.no_grad():  # Disables gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

        # Forward pass
            outputs = model(images)

        # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
            total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')



    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

