from torchvision import transforms, datasets
import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader
from model import resnet34
from tqdm import tqdm
import sys
import argparse


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device".format(device))

    data_transfrom = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    data_root = args.dataset_root
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)

    train_data = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=data_transfrom["train"]
    )
    train_num = len(train_data)

    class_idx = train_data.class_to_idx
    class_dict = dict((v, k) for k, v in class_idx.items())
    json_str = json.dumps(class_dict, indent=4)
    with open("class_index.json", "w") as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    nw = min([
        os.cpu_count(),
        batch_size if batch_size > 1 else 0,
        8]
    )
    print("Using {} dataloader workers every process".format(nw))

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )

    val_data = datasets.ImageFolder(
        root=os.path.join(data_root, "val"),
        transform=data_transfrom["val"]
    )
    val_num = len(val_data)
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw
    )

    print("Using {} for train,using {} for val".format(train_num, val_num))

    net = resnet34()
    weigth_path = args.pretrained_model
    assert os.path.exists(weigth_path), "weight file {} is not exists".format(weigth_path)
    net.load_state_dict(torch.load(weigth_path, map_location=device))

    inchannels = net.fc.in_features
    net.fc = nn.Linear(inchannels, args.num_classes)
    net.to(device) 

    params = [p for p in net.parameters() if p.requires_grad]
    learning_rate = args.lr
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    epochs = args.epoch 
    best_acc = 0.0 
    save_path = "weight/resnet34_best.pth"  
    train_step = len(train_dataloader)
    loss_r, acc_r = [], []  

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad() 
            pre = net(images.to(device))
            loss = loss_function(pre, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        loss_r.append(running_loss)

        net.eval()
        acc = 0.0 
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_d in val_bar:
                val_image, val_label = val_d
                output = net(val_image.to(device))
                predict_y = torch.max(output, dim=1)[1]
                acc += torch.eq(predict_y, val_label.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        acc_r.append(val_accurate) 
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_step, val_accurate))

        if val_accurate > best_acc: 
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("finished tarining")

    with open('training_statistic.json', 'w+') as f:
        json.dump(dict(loss=loss_r, accuracy=acc_r), f, indent=4)


def arguments():
    parser = argparse.ArgumentParser(description='Arguments for training ResNet-34')

    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='The number of categories for the network to predict'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,
        help='number of epoch'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../../data',
        help='The root directory of dataset'
    )
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default='weight/resnet34-333f7ec4.pth',
        help='The path to the network pre-training weight file'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    main(args)
