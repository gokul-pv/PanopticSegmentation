# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .custom_construction import make_construction_transforms

import logging

def box_xywh_to_xyxy(x):
    xs, ys, w, h = x.unbind(-1)
    b = [xs, ys, (xs + w), (ys + h)]
    return torch.stack(b, dim=-1)
def masks_to_boxes(segments):
    boxes = []
    labels = []
    iscrowd = []
    area = []
    for ann in segments:
        # print(ann['id'])
        if len(ann["bbox"]) == 4:
            boxes.append(ann["bbox"])
            area.append(ann['area'])
        else:
            boxes.append([0, 0, 2, 2])
            area.append(4)
        labels.append(ann["category_id"])
        iscrowd.append(ann['iscrowd'])
    if len(boxes) == 0 and len(labels) == 0:
        boxes.append([0, 0, 2, 2])
        labels.append(1)
        area.append(4)
        iscrowd.append(0)
    boxes = torch.tensor(boxes, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)
    iscrowd = torch.tensor(iscrowd)
    area = torch.tensor(area)
    boxes = box_xywh_to_xyxy(boxes)
    return boxes, labels, iscrowd, area
	
class constructionPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.construction = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.construction['images'] = sorted(self.construction['images'], key=lambda x: x['id'])
        self.construction['annotations'] = sorted(self.construction['annotations'], key=lambda x: x['image_id'])
        # sanity check
        if "annotations" in self.construction:
            for img, ann in zip(self.construction['images'], self.construction['annotations']):
                # print(img['file_name'], ann['file_name'])
                assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        # print(idx)
        ann_info = self.construction['annotations'][idx] if "annotations" in self.construction else self.construction['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')
        ann_path = Path(self.ann_folder) / ann_info['file_name']
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if "segments_info" in ann_info:
          # if len(ann_info['segments_info'])!=0 :
          masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
          masks = rgb2id(masks)

          ids = np.array([ann['id'] for ann in ann_info['segments_info']])
          masks = masks == ids[:, None, None]

          masks = torch.as_tensor(masks, dtype=torch.uint8)
          labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)
          # else:
          #   masks = None  

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        if self.return_masks:
            target['masks'] = masks
        boxes, labels, iscrowd, area = masks_to_boxes(ann_info["segments_info"])
        target['labels'] = labels

        target["boxes"] = boxes #masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['iscrowd'] = iscrowd
        target['area'] = area
        # if "segments_info" in ann_info:
        #   for name in ['iscrowd', 'area']:
        #     target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.construction['images'])

    def get_height_and_width(self, idx):
        img_info = self.construction['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width


def build(image_set, args):
    img_folder_root = Path(args.data_path)
    ann_folder_root = Path(args.data_panoptic_path)
    assert img_folder_root.exists(), f'provided DATA path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided DATA path {ann_folder_root} does not exist'
    mode = 'panoptic'
    PATHS = {
        "train": ("images", f"{mode}", f"{mode}.json"),
        "val": ("images", f"{mode}", f"{mode}_val.json"),
    }

    img_folder, ann_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder_path = ann_folder_root / ann_folder
    ann_file = ann_folder_root / ann_file

    dataset = constructionPanoptic(img_folder_path, ann_folder_path, ann_file,
                           transforms=make_construction_transforms(image_set), return_masks=args.masks)

    return dataset
