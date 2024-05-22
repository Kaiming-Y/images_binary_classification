import json
import matplotlib.pyplot as plt
import os


save_path = '../../fig/googlenet'
os.makedirs(save_path, exist_ok=True)

def plot_loss_curve() -> None:
    with open('./log/training_log.json', 'r') as f:
        log_data = json.load(f)

    train_losses = log_data['train_losses']
    test_losses = log_data['test_losses']
    train_acc = log_data['train_acc']
    test_acc = log_data['test_acc']

    plot_metrics(train_losses, test_losses, train_acc, test_acc)

    with open('./log/eval_metrics.json', 'r') as f:
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

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_plot.png'))
    plt.show()


if __name__ == '__main__':
    plot_loss_curve()
