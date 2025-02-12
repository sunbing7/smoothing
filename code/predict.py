""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse

from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime
import models

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("--dataset", help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--data_path', type=str, default='../data', help='path to the dataset')
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet50', 'MobileNetV2', 'vgg11_bn',
                             'MobileNet', 'shufflenetv2', 'densenet'])
parser.add_argument('--t_attack', type=str, default='green', help='attacked type')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument("--test_original", type=int, default=0)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

if __name__ == "__main__":
    start_time = time()
    # load the base classifier
    checkpoint = torch.load(args.base_classifier, map_location=device)
    num_class = get_num_classes(args.dataset)
    base_classifier = getattr(models, args.arch)(num_classes=num_class, pretrained=0).to(device)
    base_classifier.load_state_dict(checkpoint)

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset, dataset_adv = get_dataset(args.dataset, args.split, args.data_path, args.t_attack, args.poison_target)

    #test clean samples
    n_total = 0
    n_correct = 0
    n_ori_correct = 0
    base_classifier.eval()
    print('total number of clean samples: {}'.format(len(dataset)))
    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        before_time = time()

        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
        if args.test_original:
            input = x[None,:]
            ori_predict = base_classifier(input).data.max(1)[1]

        after_time = time()
        correct = int(prediction == label)
        if correct:
            n_correct = n_correct + 1
        if args.test_original:
            if ori_predict == label:
                n_ori_correct = n_ori_correct + 1

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)
        n_total = n_total + 1
    print('accuracy:{}'.format(n_correct / n_total))
    if args.test_original:
        print('original accuracy:{}'.format(n_ori_correct / n_total))

    #test attacked samples
    n_total = 0
    n_correct = 0
    n_ori_correct = 0
    base_classifier.eval()
    print('total number of attacked samples: {}'.format(len(dataset_adv)))
    for i in range(0, len(dataset_adv)):
        (x, label) = dataset_adv[i]
        x = x.cuda()
        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
        if args.test_original:
            input = x[None,:]
            ori_predict = base_classifier(input).data.max(1)[1]

        after_time = time()
        correct = int(prediction == label)
        if correct:
            n_correct = n_correct + 1
        if args.test_original:
            if ori_predict == label:
                n_ori_correct = n_ori_correct + 1
        n_total = n_total + 1
    print('SR: {}'.format(n_correct / n_total))
    if args.test_original:
        print('original SR:{}'.format(n_ori_correct / n_total))
    f.close()
    execution_time = time() - start_time
    print('{} Running time: {}s'.format(args.base_classifier, execution_time))
