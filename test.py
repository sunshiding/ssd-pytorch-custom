"""Test script to evaluate trained SSD Multibox Detector model
To run, e.g.:  
python test.py --trained_model weights\Custom.pth --dataset_root data\image_data\test --visual_threshold 0.3
"""

from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import CUSTOM_ROOT, CUSTOM_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import CustomAnnotationTransform, CustomDetection, BaseTransform, CUSTOM_CLASSES
from data.config import custom, MEANS
from utils.augmentations import SSDAugmentation


import torch.utils.data as data
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=VOC_ROOT, help='Location of data root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

# Check for CUDA/GPU support and warn if it does exist
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = os.path.join(os.getcwd(), args.save_folder)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# If GPU is available use it, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = os.path.join(save_folder, 'test1.txt')
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img, annotation, _, _, img_id = testset.pull_item(i)
        print(img_id)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x = img.unsqueeze(0)
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = Variable(x)

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')

        net = net.to(device)

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= args.visual_threshold:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = CUSTOM_CLASSES[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                print(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords))
                j += 1


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.dataset_root, [('2007', 'test')], None, VOCAnnotationTransform())
    # GPU support if available
    net = net.to(device)
    if args.cuda:
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

def test_custom():
    cfg = custom

    # net = build_ssd('test', 300, num_classes) # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model))

    # load net
    net = build_ssd(phase='test', size=cfg['min_dim'], num_classes=cfg['num_classes'])
    net.load_state_dict(torch.load(args.trained_model))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = net.to(device)
    
    net.eval()
    print('Finished loading model!')
    # load data
    testset = CustomDetection(root=args.dataset_root, 
                                image_set=[('test')], 
                                transform=BaseTransform(cfg['min_dim'], MEANS), 
                                target_transform=CustomAnnotationTransform(train=False))

    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)



if __name__ == '__main__':
    test_custom()
