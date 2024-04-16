import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from model import DeepHybridnet  
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root='G:/vineet/medicinal_plant/Code/V2/folio_data/test', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assuming num_classes is known or can be inferred
    num_classes = 32
    model = DeepHybridnet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('G:/vineet/medicinal_plant/Code/V2/weights/mode_folio2.pth', map_location=device))
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds )
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(25, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
if __name__ == '__main__':
    main()


