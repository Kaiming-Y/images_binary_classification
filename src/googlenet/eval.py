import argparse
import os
import json
from argparse import Namespace
import torch
from data_processor import load_data
from model import googlenet
from utils import calculate_metrics


def main(args):
    # Parameters
    batch_size = args.batch_size
    data_dir = os.path.join(args.dataset_dir, 'dataset')
    augment_dir = os.path.join(args.dataset_dir, 'dataset_augmented')
    model_path = os.path.join(args.model_dir, args.model)
    log_file = './log/eval_metrics.json'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Load data
    _, test_loader = load_data(data_dir, augment_dir, batch_size)

    # Init GoogLeNet
    model = googlenet(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluation
    accuracy, precision, recall, f1_score = calculate_metrics(model, test_loader, device)

    # Print result
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    # Save the metrics
    metrics_data = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }
    with open(log_file, 'w') as f:
        json.dump(metrics_data, f)


def arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Arguments for evaluating GoogLeNet')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size, the general setting range is between 4 and 32.')
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='../../data',
                        help='Directory to the datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../model/googlenet',
                        help='Directory to the model weights')
    parser.add_argument('--model',
                        type=str,
                        default='googlenet_model.pth',
                        help='Name of the pre-trained model in model directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    main(args)
