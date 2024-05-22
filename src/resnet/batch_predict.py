import os
import json
import argparse
import shutil
import torch
from PIL import Image
from torchvision import transforms
from glob import glob
from pathlib import Path

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

    lower_conf_preds, wrong_preds = [], set()
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
                if pro < args.conf_thr:
                    lower_conf_preds.append([
                        img_path_list[ids * batch_size + idx],
                        class_indict[str(cla.numpy())],
                        float(pro.numpy())
                    ])

            batch_predicted_clses = classes.numpy().tolist()
            FP = [i for i, (g, p) in enumerate(zip(batch_ground_truths, batch_predicted_clses))
                  if g == 0 and p == 1]
            FN = [i for i, (g, p) in enumerate(zip(batch_ground_truths, batch_predicted_clses))
                  if g == 1 and p == 0]
            for i in FP + FN:
                wrong_preds.add(img_path_list[start + i])

    if os.path.exists('low_probs/'):
        shutil.rmtree('low_probs/')
    os.makedirs('low_probs/')
    for low_conf_pred in lower_conf_preds:
        shutil.copyfile(
            low_conf_pred[0],
            os.path.join('low_probs',
                         f'{Path(low_conf_pred[0]).stem}_{low_conf_pred[1]}_{low_conf_pred[2]:.4f}.png')
        )
    print(f"All images with a confidence level lower than {args.conf_thr} are saved to the 'low_probs' directory.")


    print(f'The following are incorrectly classified pictures:')
    for wrong_pred in wrong_preds:
        print(wrong_pred)


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='The number of categories that the network needs to classify'
    )
    parser.add_argument(
        '--img_paths',
        type=str,
        default='data/val/*/*.png',
        help='Path to test set images'
    )
    parser.add_argument(
        '--cls_index',
        type=str,
        default='class_index.json',
        help='The path to the json file that defines category information'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        default='weight/resnet34_best.pth',
        help='The path of the weight file that needs to be loaded. The weight file obtained after network training should be used here.'
    )
    parser.add_argument(
        '--conf_thr',
        type=float,
        default=1.,
        help='Used to filter image classification results below this threshold. The recommended setting range is 0.5~1.0'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size. It can be dynamically adjusted to an exponential multiple of 2 (such as 2/4/8/16...) according to the hardware/software conditions of your computer.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    main(args)
