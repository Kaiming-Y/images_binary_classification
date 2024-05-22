import os
import random
import shutil
import argparse

from glob import glob
from pathlib import Path
from tqdm import tqdm


def get_arguments():
    arg_parser = argparse.ArgumentParser(description='Data set partitioning parameter settings')
    arg_parser.add_argument('--dataset_root',
                            default='..\..\data',
                            type=str,
                            help='Dataset root directory')
    arg_parser.add_argument('--train_ratio',
                            default=0.8,
                            type=float,
                            help='The training set ratio is 0.8 by default, and the corresponding test set ratio is 0.2')
    return arg_parser.parse_args()


def processor(dataset_root: str,
              train_ratio: float = 0.8):
    """
    Process data and split them into trainset and testset.
    :param dataset_root: Directory to the dataset.
    :param train_ratio: Ratio of the train set regarding to the entire dataset.
    :return: None
    """ 
    assert os.path.exists(dataset_root) and os.path.isdir(dataset_root), \
        'Invalid dataset root directory!'
    assert 0.5 < train_ratio < 1, 'Invalid trainset ratio!' 

    neg_samples = glob(os.path.join(dataset_root, 'sea/*.png'), recursive=False)
    random.shuffle(neg_samples)
    pos_samples = glob(os.path.join(dataset_root, 'ship/*.png'), recursive=False)
    random.shuffle(pos_samples)
    num_neg_samples = len(neg_samples)
    num_pos_samples = len(pos_samples)

    num_neg_training_samples = round(num_neg_samples * train_ratio) 
    neg_training_samples = random.sample(neg_samples, num_neg_training_samples)
    neg_testing_samples = [x for x in neg_samples if x not in neg_training_samples] 

    num_pos_training_samples = round(num_pos_samples * train_ratio)
    pos_training_samples = random.sample(pos_samples, num_pos_training_samples)
    pos_testing_samples = [x for x in pos_samples if x not in pos_training_samples]

    train_dir = os.path.join(dataset_root, 'train/')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(os.path.join(train_dir, 'sea/')) 
    os.makedirs(os.path.join(train_dir, 'ship/'))
    for neg_train_sample in neg_training_samples: 
        shutil.copyfile(
            src=neg_train_sample,
            dst=os.path.join(dataset_root, f'train/sea/{Path(neg_train_sample).name}')
        )
    for pos_train_sample in pos_training_samples:
        shutil.copyfile(
            src=pos_train_sample,
            dst=os.path.join(dataset_root, f'train/ship/{Path(pos_train_sample).name}')
        )

    test_dir = os.path.join(dataset_root, 'val/')
    if os.path.exists(test_dir): 
        shutil.rmtree(test_dir)
    os.makedirs(os.path.join(test_dir, 'sea/'))
    os.makedirs(os.path.join(test_dir, 'ship/'))
    for neg_test_sample in neg_testing_samples:
        shutil.copyfile(
            src=neg_test_sample,
            dst = os.path.join(dataset_root, f'val/sea/{Path(neg_test_sample).name}')
        )
    for pos_test_sample in pos_testing_samples:
        shutil.copyfile(
            src=pos_test_sample,
            dst=os.path.join(dataset_root, f'val/ship/{Path(pos_test_sample).name}')
        )


if __name__ == "__main__":
    arguments = get_arguments()
    processor(dataset_root=arguments.dataset_root,
              train_ratio=arguments.train_ratio)
