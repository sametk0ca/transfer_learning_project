import argparse
import torch
import torchvision
import os
from torchvision import datasets, transforms, models

import tarfile
import urllib.request


def load_data():
    # dataset download
    root_dir = './data'
    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    filename = 'lfw.tgz'
    filepath = os.path.join(root_dir, filename)
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    if not os.path.exists(filepath):
        # URL download
        urllib.request.urlretrieve(url, filepath)
        
        # unzip
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(root_dir)
        
        # remove
        os.remove(filepath)
        
    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader



# Save model parameters
def save_model(model, model_name, scenario):
    path = f"weight/{model_name}_Scenario{scenario}.pt"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



def replace_output_layer(model, num_classes):
    print(type(model))
    if isinstance(model, models.AlexNet):
        last_linear = None
        for layer in model.classifier:
            if isinstance(layer, torch.nn.Linear):
                last_linear = layer
        if last_linear is None:
            raise ValueError("No linear layer found in the classifier")
        num_features = last_linear.in_features
        model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

    elif isinstance(model, models.ResNet):
        
        model.fc = torch.nn.Linear(2048, num_classes)  # Update the last linear layer

    
    elif isinstance(model, models.VGG):
        last_linear = None
        for layer in model.classifier:
            if isinstance(layer, torch.nn.Linear):
                last_linear = layer
        if last_linear is None:
            raise ValueError("No linear layer found in the classifier")
        num_features = last_linear.in_features
        model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
    
    elif isinstance(model, models.ResNet):
        # Find the average pooling layer
        avg_pool_layer = None
        for module in model.children():
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                avg_pool_layer = module
                break
        if avg_pool_layer is None:
            raise ValueError("No average pooling layer found in the model")
        
        # Find the last linear layer after average pooling
        last_linear = None
        for module in model.children():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("No linear layer found after average pooling")
        num_features = last_linear.in_features
        last_linear = torch.nn.Linear(num_features, num_classes)
    
    else:
        raise NotImplementedError("Model not supported")
    return model 






def feature_extraction(model, train_loader, test_loader, num_classes):
    # Prune fully connected layers
    if isinstance(model, torchvision.models.AlexNet):
        model.classifier = model.classifier[:6]
    elif isinstance(model, torchvision.models.ResNet):
        model.fc = torch.nn.Sequential(*list(model.children())[:-1])
    elif isinstance(model, torchvision.models.VGG):
        model.classifier = model.classifier[:6]
    else:
        raise NotImplementedError("Model not supported")

    # Replace output layer
    model = replace_output_layer(model, num_classes)

    # Move model to CUDA if available ( i used nvidia rtx 3060 )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Feature extraction
    model.eval()
    with torch.no_grad():
        train_features = []
        train_labels = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to CUDA if available ( i used nvidia rtx 3060 )
            features = model(images)
            train_features.append(features)
            train_labels.append(labels)

        test_features = []
        test_labels = []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to CUDA if available ( i used nvidia rtx 3060 )
            features = model(images)
            test_features.append(features)
            test_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    return train_features, train_labels, test_features, test_labels


def fine_tuning(model, train_loader, test_loader, num_classes, freeze_features=True, unfreeze_classifier=True):
    
    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.bypass_train:
        # Load pre-trained weights
        model.load_state_dict(torch.load(f"weight/{args.model}_Scenario{args.scenario}.pt"))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to CUDA if available (i used nvidia rtx 3060 gpu to train)
                
                if len(images.shape) == 3:  # If input data is 3D (batch size, height, width)
                    images = images.unsqueeze(1)  # Add channel dimension
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Accuracy: {accuracy}")
    else:
        if freeze_features:
            
            if isinstance(model, torchvision.models.AlexNet):
                for param in model.features.parameters():
                    param.requires_grad = False
            elif isinstance(model, torchvision.models.ResNet):
                for param in model.parameters():
                    param.requires_grad = False
            elif isinstance(model, torchvision.models.VGG):
                for param in model.parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError("This model is not supported for fine-tuning") 

        if unfreeze_classifier:
            
            if isinstance(model, torchvision.models.AlexNet):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif isinstance(model, torchvision.models.ResNet):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif isinstance(model, torchvision.models.VGG):
                for param in model.classifier[5].parameters():
                    param.requires_grad = True
            else:
                raise NotImplementedError("This model is not supported for fine-tuning") 

        
        # Fine-tuning
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        
        num_epochs = 5
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to CUDA if available (i used nvidia rtx 3060 gpu to train)

                if len(images.shape) == 3:  # If input data is 3D (batch size, height, width)
                    images = images.unsqueeze(1)  # Add channel dimension

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to CUDA if available (i used nvidia rtx 3060 gpu to train)
                
                if len(images.shape) == 3:  # If input data is 3D (batch size, height, width)
                    images = images.unsqueeze(1)  # Add channel dimension
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy: {accuracy}")



def main(args):
    # Load data
    train_loader, test_loader = load_data()

    # Define model
    if args.model == "AlexNet":
        model = models.alexnet(weights=True)
    elif args.model == "ResNet":
        model = models.resnet50(weights=True)
    elif args.model == "VGG":
        model = models.vgg16(weights=True)
    else:
        raise NotImplementedError("Model not supported")

    num_classes = len(train_loader.dataset.dataset.classes)

    # Perform feature extraction or fine-tuning based on scenario
    if args.scenario == 1:
        # Replace the output layer to match the number of classes in the new dataset
        model = replace_output_layer(model, num_classes)
    elif args.scenario == 2:
        # Extract features
        train_features, train_labels, test_features, test_labels = feature_extraction(model, train_loader, test_loader, num_classes)
    elif args.scenario == 3:
        # Fine-tuning by disregarding the weight parameters at very late CNN blocks and learning them
        train_features, train_labels, test_features, test_labels = feature_extraction(model, train_loader, test_loader, num_classes)
        fine_tuning(model, train_loader, test_loader, num_classes)
    elif args.scenario == 4:
        # Fine-tuning by training the whole network
        train_features, train_labels, test_features, test_labels = feature_extraction(model, train_loader, test_loader, num_classes)
        fine_tuning(model, train_loader, test_loader, num_classes)
    else:
        raise ValueError("Invalid sceario number")

    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning for LFW Dataset")

    # --bypass_train  
    parser.add_argument("--bypass_train", action="store_true", help="Bypass the training phase")

    # --model and --scenario  
    parser.add_argument("--model", choices=["AlexNet", "ResNet", "VGG"], default="AlexNet", help="Choose the model (default: AlexNet)")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], default=1, help="Choose the scenario (default: 1)")

    args = parser.parse_args()

    main(args)

#210717021 Samet Koca