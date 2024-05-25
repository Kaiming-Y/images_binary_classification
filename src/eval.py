import argparse
import os
import json
from argparse import Namespace
import torch
from data_loader import load_data
from utils import calculate_metrics
from datetime import datetime
from models import get_model


def main(args):
    log_file = None

    try:
        # Parameters
        num_classes = args.num_classes
        batch_size = args.batch_size
        data_dir = os.path.join(args.data_dir, 'dataset')
        augment_dir = os.path.join(args.data_dir, 'dataset_augmented')
        weight_root_dir = args.weight_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'./log/{args.model}_eval_metrics_augmented_{timestamp}.json' if args.augment else f'./log/{args.model}_eval_metrics_{timestamp}.json'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Load data
        _, test_loader = load_data(data_dir, augment_dir, batch_size, 0.2, args.augment)

        # Init model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(args.model, num_classes)
        trained_model_file = os.path.join(weight_root_dir, f'{args.model}_model.pth')
        assert os.path.exists(trained_model_file), f"weight file {trained_model_file} does not exist"
        model.load_state_dict(torch.load(trained_model_file, map_location=device))
        model.to(device)

        start_time = time.time()

        # Evaluation
        accuracy, precision, recall, f1_score = calculate_metrics(model, test_loader, device)

        end_time = time.time()
        test_time = end_time - start_time

        # Print result
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1_score:.4f}')
        print(f'Test Time: {test_time:.4f} seconds')

        # Save the metrics
        metrics_data = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'test_time': test_time
        }
        with open(log_file, 'w') as f:
            json.dump(metrics_data, f)

    except Exception as e:
        print(f"An error occurred: {e}")
        if log_file and os.path.exists(log_file):
            os.remove(log_file)
        raise
    finally:
        if log_file and os.path.exists(log_file):
            print(f"Log file generated: {log_file}")
        else:
            print("Log file was removed due to an error.")


def arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Arguments for evaluating models')

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        choices=['googlenet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                                 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                 'resnet152'],
                        help='The name of the model to evaluate')
    parser.add_argument('--augment',
                        action='store_true',
                        help='Whether to augment the data or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='The number of categories for the network to predict')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size, the general setting range is between 4 and 32.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../data',
                        help='Directory to the datasets')
    parser.add_argument('--weight_dir',
                        type=str,
                        default='../model_weight',
                        help='Directory to the model weights')
    # parser.add_argument('--trained_model',
    #                     type=str,
    #                     default=None,
    #                     required=True,
    #                     help='Name of the trained model in model weights directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    main(args)
