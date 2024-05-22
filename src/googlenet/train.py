import argparse
from argparse import Namespace
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from data_processor import load_data
from model import googlenet


def main(args):
    # Hyperparameters & init device
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epoch
    num_classes = args.num_classes
    data_dir = os.path.join(args.dataset_dir, 'dataset')
    augment_dir = os.path.join(args.dataset_dir, 'dataset_augmented')
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    log_file = './log/training_log.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, test_loader = load_data(data_dir, augment_dir, batch_size)

    # Init googlenet model
    model = googlenet(num_classes=num_classes)
    if args.pretrained_model is not None:
        pretrained_model_dir = os.path.join(model_dir, args.pretrained_model)
        assert os.path.exists(pretrained_model_dir), "weight file {} is not exists".format(pretrained_model_dir)
        model.load_state_dict(torch.load(pretrained_model_dir, map_location=device))
    model.to(device)

    # Loss function & Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train googlenet model
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = running_loss / len(test_loader)
        test_accuracy = 100.0 * correct / total
        test_losses.append(test_loss)
        test_acc.append(test_accuracy)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}],'
            f' Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,'
            f' Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        # Save the weights of googlenet if test accuracy is higher than the best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(model_dir, f'googlenet_model.pth'))

        # Save loss and accuracy to log
        log_data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_acc': train_acc,
            'test_acc': test_acc,
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f)


def arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Arguments for training GoogLeNet')

    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='The number of categories for the network to predict')
    parser.add_argument('--epoch',
                        type=int,
                        default=200,
                        help='The number of Epochs for network training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size, the general setting range is between 4 and 32.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Network training learning rate')
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='../../data',
                        help='Directory to the datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../model/googlenet',
                        help='Directory to the model weights')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='Name of the pre-trained model in model directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    main(args)
