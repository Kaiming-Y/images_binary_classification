import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
from glob import glob

from model import resnet34


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_paths = args.img_paths
    img_path_list = glob(test_paths, recursive=True)

    json_path = args.cls_index
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    json_file = open(json_path, "r")
    class_indict = json.load(json_file) 
    class_indict_reverse = {v: k for k, v in class_indict.items()} 

    ground_truths = [int(class_indict_reverse[x.split('/')[-2]])
                     for x in img_path_list]

    model = resnet34(num_classes=args.num_classes).to(device)

    weight_path = args.weight_path
    assert os.path.exists(weight_path), f"file: '{weight_path}' dose not exist."
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval() 
    batch_size = args.batch_size
    TPs, TNs, FPs, FNs = 0, 0, 0, 0 

    with torch.no_grad(): 

        for ids in range(0, round(len(img_path_list) / batch_size)): 
            img_list = []
            start = ids * batch_size
            end = -1 if (ids + 1) * batch_size >= len(img_path_list) else (ids + 1) * batch_size
            for img_path in img_path_list[start: end]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)
            batch_ground_truths = ground_truths[start: end]
            batch_img = torch.stack(img_list, dim=0)

            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))

            batch_predicted_clses = classes.numpy().tolist()

            TP = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == p == 1])
            TN = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == p == 0])
            FP = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == 0 and p == 1])
            FN = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == 1 and p == 0])

            TPs += TP
            TNs += TN
            FPs += FP
            FNs += FN

    accuracy = (TNs + TPs) / len(img_path_list)
    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'Overall performance:\n'
          f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FPS: {fps:.2f}')


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='The number of categories for the network to predict'
    )
    parser.add_argument(
        '--img_paths',
        type=str,
        default='data/val/*/*.png',
        help='Matching addresses of all images to be tested. The asterisk * represents a matching character. The program will match matching files based on this path, and finally load it into a file path list.'
    )
    parser.add_argument(
        '--cls_index',
        type=str,
        default='class_index.json',
        help='Path to the class_index.json file generated during the training phase'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        default='weight/resnet34_best.pth',
        help='Saved optimal model weights'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size, the recommended setting range is 4~32 (multiples of 2)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    main(args)
