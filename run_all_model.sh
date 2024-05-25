#!/bin/bash

cd ./src

LOG_FILE="../training_evaluation.log"
models=('googlenet' 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')

> $LOG_FILE

# Train and eval every model
for model in "${models[@]}"
do
    echo "Training and evaluating model: $model" | tee -a $LOG_FILE
    
    python train.py --model $model 2>&1 | tee -a $LOG_FILE
    python eval.py --model $model 2>&1 | tee -a $LOG_FILE
    
    python train.py --model $model --augment 2>&1 | tee -a $LOG_FILE
    python eval.py --model $model --augment 2>&1 | tee -a $LOG_FILE
    
    echo "Completed training and evaluating model: $model" | tee -a $LOG_FILE
done

echo "All models have been trained and evaluated." | tee -a $LOG_FILE
