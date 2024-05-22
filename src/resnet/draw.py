import os
import json
import argparse
import matplotlib.pyplot as plt

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',
                        default='./training_statistic.json',
                        type=str,
                        help='The path to the training_statistic.json file generated during training. It stores the loss and accuracy data during network training.')
    parser.add_argument('--save_dir',
                        default='./',
                        type=str,
                        help='Store the path for drawing the curve graph')
    return parser.parse_args()

def draw_plots(json_path: str,
               save_dir: str):

    with open(json_path, 'r') as f:
        statistics = json.load(f)

    loss, accuracy = statistics['loss'], statistics['accuracy']

    plt.figure(1)
    plt.plot(range(len(loss)), loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve of training')
    plt.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=600)
    plt.show()

    plt.figure(2)
    plt.plot(range(len(accuracy)), accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve of training')
    plt.savefig(os.path.join(save_dir, 'train_accuracy.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    args = get_arguments()
    draw_plots(args.json_path, args.save_dir)