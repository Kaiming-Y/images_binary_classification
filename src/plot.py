import argparse
from argparse import Namespace
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np


def plot(args) -> None:
    train_log_file = args.log_file

    with open(train_log_file, 'r') as f:
        log_data = json.load(f)

    train_losses = log_data['train_losses']
    test_losses = log_data['test_losses']
    train_acc = log_data['train_acc']
    test_acc = log_data['test_acc']
    test_times = log_data['test_times']

    plot_loss_acc(train_losses, test_losses, train_acc, test_acc, args)
    plot_test_times(test_times, args)


def plot_loss_acc(train_losses, test_losses, train_acc, test_acc, args) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.plot(epochs, test_losses, 'b-', label='Test Loss')
    plt.title(f'Loss ({args.model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Train Accuracy')
    plt.plot(epochs, test_acc, 'b-', label='Test Accuracy')
    plt.title(f'Accuracy ({args.model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f'{args.model_name}_loss_acc_{timestamp}.png'))
    plt.show()


def plot_test_times(test_times, args):
    epochs = range(1, len(test_times) + 1)
    avg_test_time = np.mean(test_times)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, test_times, 'g-', label='Test Time')
    plt.axhline(y=avg_test_time, color='r', linestyle='--', label=f'Avg Time: {avg_test_time:.4f} sec')
    plt.title(f'Test Time per Epoch ({args.model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()

    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f'{args.model_name}_test_time_{timestamp}.png'))
    plt.show()


def arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Arguments for GoogLeNet plotting')

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        choices=['googlenet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                                 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                 'resnet152'],
                        help='The name of the model to train')
    parser.add_argument('--log_file',
                        type=str,
                        required=True,
                        help='The path to training log file')
    parser.add_argument('--output_dir',
                        type=str,
                        default='../fig',
                        help='Directory to save the figure')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    plot(args)
