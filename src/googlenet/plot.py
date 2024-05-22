import argparse
from argparse import Namespace
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np


def plot_loss_curve(args) -> None:
    with open(args.train_log, 'r') as f:
        log_data = json.load(f)

    train_losses = log_data['train_losses']
    test_losses = log_data['test_losses']
    train_acc = log_data['train_acc']
    test_acc = log_data['test_acc']
    test_times = log_data['test_times']

    plot_metrics(train_losses, test_losses, train_acc, test_acc)
    plot_test_times(test_times)

    with open(args.eval_metric, 'r') as f:
        metrics_data = json.load(f)

    accuracy = metrics_data['accuracy']
    precision = metrics_data['precision']
    recall = metrics_data['recall']
    f1_score = metrics_data['f1_score']

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')


def plot_metrics(train_losses, test_losses, train_acc, test_acc) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'r-', label='Train Loss')
    plt.plot(test_losses, 'b-', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'r-', label='Train Accuracy')
    plt.plot(test_acc, 'b-', label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f'loss_and_accuracy_plot_{timestamp}.png'))
    plt.show()


def plot_test_times(test_times):
    epochs = range(1, len(test_times) + 1)
    avg_test_time = np.mean(test_times)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, test_times, 'g-', label='Test Time')
    plt.axhline(y=avg_test_time, color='r', linestyle='--', label=f'Avg Time: {avg_test_time:.4f} sec')
    plt.title('Test Time per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f'test_time_plot_{timestamp}.png'))
    plt.show()


def arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Arguments for GoogLeNet plotting')

    parser.add_argument('--train_log',
                        type=str,
                        default='./log/training_log.json',
                        help='The path to training log')
    parser.add_argument('--eval_metric',
                        type=str,
                        default='./log/eval_metrics.json',
                        help='The path to evaluation metrics')
    parser.add_argument('--save_dir',
                        type=str,
                        default='../../fig/googlenet',
                        help='Directory to save the figure')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    plot_loss_curve(args)
