from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import csv
import json
import glob
import skimage

CUSTOM_ROOT = os.path.join(HOME, 'data', 'image_data')
print(HOME)
# Classes do not explicitley have BG here
CUSTOM_CLASSES = ('object')

def get_targets(label_file):
    """
    Processes the VGG Image Annotator json output into a list of dicts.

    Args:
        label_file : file (json)
            Json annotation file snippet:
            {'480835.jpg1406946': {'regions': {}, 'filename': '480835.jpg', 'fileref': '', 
            'size': 1406946, 'file_attributes': {}, 'base64_img_data': ''}, 
            '480980.jpg2644207': {'regions': {'0': {'region_attributes': {}, 'shape_attributes': 
            {'y': 547, 'x': 2176, 'height': 249, 'name': 'rect', 'width': 178}}}, 'filename': 
            '480980.jpg', 'fileref': '', 'size': 2644207, 'file_attributes': {}, 'base64_img_data': ''},...

    For id we will use is the filename
    regions contains the shape attributes (might be many regions/many bounding boxes for an image)
    shape_attributes have the bounding boxes in this json

    Returns:
        dict with the bounding box(es) for an id (the filename in this instance)
    """
    targets = {}
    # Shape regions json (exported from VGG Image Annotator)
    with open(label_file, 'r') as f:
        json_data = json.load(f)
        for key in json_data:
            file_data = json_data[key]
            if file_data['regions'] != {}:
                target_id = file_data['filename']
                # IDS.append()
                region_data = file_data['regions']
                bboxes = []
                for bbox_cnt in region_data: # go through bounding boxes (may be multiple)
                    shape_data = region_data[bbox_cnt]['shape_attributes']
                    bboxes.append(shape_data)
                targets[target_id] = bboxes

    # Return, e.g.:
    # {'filename/id' : [{'y': 547, 'x': 2176, 'height': 249, 'name': 'rect', 'width': 178}, ...]}
    return targets


class CustomAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, train=True):
        if train == True:
            self.targets = get_targets(os.path.join(CUSTOM_ROOT, 'train', 'annot', 'via_region_data.json'))
        else:
            self.targets = get_targets(os.path.join(CUSTOM_ROOT, 'test', 'annot', 'via_region_data.json'))
        self.ids = list(self.targets.keys())

    def __call__(self, target, width, height):
        """
        Args:
            target (list): list of dicts, annotations
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes and class ids  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        custom_class = 0 # int for not background (BG)
        for _, elem in enumerate(target):
            bbox = np.zeros(shape=4)
            bbox[0] = elem['x']
            bbox[1] = elem['y']
            bbox[2] = bbox[0] + elem['width']
            bbox[3] = bbox[1] + elem['height']
            # TODO understand purpose of scale
            # final_box = list(np.array(bbox)/scale)
            final_box = list(np.array(bbox))
            final_box.append(custom_class)
            res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class CustomDetection(data.Dataset):
    """`Custom Detection - data can be any set of images with labels (bboxes and annotation).
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set=None, transform=None,
                 target_transform=CustomAnnotationTransform(), dataset_name='CUSTOM',
                 targets=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ids = target_transform.ids
        self.name = dataset_name
        self.targets = target_transform.targets
        self.root = os.getcwd() + os.sep + root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w, _ = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width, image_id).
        """

        img_id = self.ids[index]
        target = self.targets[img_id]

        path = os.path.join(self.root, img_id)
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(os.path.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, img_id

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]

        path = os.path.join(self.root, img_id)
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)

        return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''

        img_id = self.ids[index]
        target = self.targets[img_id]

        path = os.path.join(self.root, img_id)
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(os.path.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        boxes, labels = target[:, :4], target[:, 4]

        # Gather all annotations and bounding boxes from image
        annot_boxes = []
        for i, bbox in enumerate(boxes):
            annot_boxes.append((labels[i], bbox))
        
        return img_id, annot_boxes

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
