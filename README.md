# Image Binary Classification

This project focuses on the task of binary image classification using custom implementations of various deep learning models with PyTorch. The models used in this project include variants of VGG, GoogLeNet, and ResNet.

## Models Implemented

The following models are implemented in this project:
- **GoogLeNet**
- **VGG**: 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
- **ResNet**: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'

Each model has been carefully crafted to follow the original architecture specifications as closely as possible.

## Dataset

The dataset used in this project consists of 4000 images related to ships. It includes:
- 3000 images of non-ship objects (sea)
- 1000 images of ships

The dataset is divided into training and testing sets to evaluate the performance of the models on the binary classification task.

## Project Features

### 1. Model Training and Evaluation
- **Training**: Custom training loops are implemented to train each model on the binary classification task. The training process includes monitoring of loss and accuracy.
- **Evaluation**: Each model is evaluated on a test dataset, and performance metrics such as accuracy, precision, recall, and F1-score are calculated.

### 2. Visualization
- **Loss and Accuracy Curves**: Training and validation loss and accuracy are plotted for each model to visualize the learning process.
- **Inference Time**: The average inference time (model testing time) is calculated and visualized to compare the efficiency of different models.

### 3. Performance Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

These metrics are used to comprehensively evaluate the performance of each model on the binary classification task.

## Installation and Usage

### Installation

#### Conda

``` shell
conda env create -f environment.yml
```

#### Pip

```shell
pip install requirement.txt
```

### Usage

Enter the `src` folder.

```shell
cd ./src
```

#### Data Preprocessing

To augment the data set, randomly rotate the pictures, augment 5000 pictures of ships in the original data set, and augment 3000 pictures of non-ships. The final data set was obtained with a total of 12,000 pictures (6,000 pictures each for ships and non-ships)

```shell
python data_augmentation.py
```

#### Train & Evaluate models

1. Training

   If you want to use augmented data:

   ```shell
   python train.py --model googlenet --augment --epoch 100
   ```

   else

   ```shell
   python train.py --model googlenet --epoch 100
   ```

   You can choose a model from the [models implemented](## Models Implemented) following the argument `--model`.

2. Evaluation

   Evaluating a model you have trained successfully.

   ```shell
   python eval.py --model googlenet --augment
   ```

More information about arguments can be found in `train.py` and `eval.py`.

If you want to train and evaluate all the models, you can run the shell scripts `run_all_model.sh` to do so. Besides, you can also run `train_all.sh` to train all the model or run `eval_all.sh` to evaluate all the model (if you have trained them already).

### Visualization

Plot the loss function curve, accuracy curve and inference time.

```shell
python plot.py --model googlenet --log_file /PATH/TO/TRAINING_LOG_FILE
```

The default directory to save the output figure is `/fig`. 

## References

- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- GoogLeNet: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Acknowledgements

- [PyTorch](https://pytorch.org/) - The deep learning framework used for this project.
- Original papers and authors of VGG, GoogLeNet, and ResNet for their groundbreaking work in deep learning.
