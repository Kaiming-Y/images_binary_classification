import argparse
from argparse import Namespace
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from data_processor import load_data
from model import VGG, GoogLeNet, ResNet
from datetime import datetime
import time


def main(args):
    # Hyperparameters & init device
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epoch
    num_classes = args.num_classes
    data_dir = os.path.join(args.dataset_dir, 'dataset')
    augment_dir = os.path.join(args.dataset_dir, 'dataset_augmented')
    model_dir = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'./log/{args.model_name}_training_log_{timestamp}.json'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, test_loader = load_data(data_dir, augment_dir, batch_size, 0.2, args.augment)

    # Init model
    if args.model_name == 'googlenet':
        model = GoogLeNet.googlenet(num_classes=num_classes)
    elif args.model_name == 'vgg':
        model = VGG.vgg13(num_classes=num_classes)
    elif args.model_name == 'resnet':
        model = ResNet.resnet(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model name: {args.model_name}')

    if args.pretrained_model is not None:
        pretrained_model_dir = os.path.join(model_dir, args.pretrained_model)
        assert os.path.exists(pretrained_model_dir), f"weight file {pretrained_model_dir} does not exist"
        model.load_state_dict(torch.load(pretrained_model_dir, map_location=device))
    model.to(device)

    # Loss function & Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation loop
    train_losses, test_losses, train_acc, test_acc, test_times = [], [], [], [], []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

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
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        end_time = time.time()
        test_time = end_time - start_time
        test_times.append(test_time)

        test_loss = running_loss / len(test_loader)
        test_accuracy = 100.0 * correct / total
        test_losses.append(test_loss)
        test_acc.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,'
              f' Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Time: {test_time:.4f} seconds')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(model_dir, f'{args.model_name}_model.pth'))

        log_data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_times': test_times
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f)


def arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Arguments for training models')

    parser.add_argument('--model_name',
                        type=str,
                        default='googlenet',
                        choices=['googlenet', 'vgg', 'resnet'],
                        help='The name of the model to train')
    parser.add_argument('--augment',
                        action='store_true',
                        help='Whether to augment the data or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='The number of categories for the network to predict')
    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help='The number of Epochs for network training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size, the general setting range is between 4 and 32.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Network training learning rate')
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='../data',
                        help='Directory to the datasets')
    parser.add_argument('--model_weight_dir',
                        type=str,
                        default='../model_weight',
                        help='Directory to the model weights')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='Name of the pre-trained model in model directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    main(args)
