#!/bin/bash

cd ./src || { echo "Failed to change directory to ./src"; exit 1; }

LOG_FILE="../evaluation.log"
models=('googlenet' 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')

> $LOG_FILE

# Train and eval every model
for model in "${models[@]}"
do
    echo "Evaluating model: $model" | tee -a $LOG_FILE
    
    python eval.py --model $model 2>&1 | tee -a $LOG_FILE
    python eval.py --model $model --augment 2>&1 | tee -a $LOG_FILE
    
    echo "Completed evaluating model: $model" | tee -a $LOG_FILE
done

echo "All models have been evaluated." | tee -a $LOG_FILE
