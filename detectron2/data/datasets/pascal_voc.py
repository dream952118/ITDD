# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import shuffle

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_pascal_voc"]


# fmt: off
CLASS_NAMES = [
	'I-Others','E-M1-Al Residue',
	'I-Dust','T-ITO1-Hole','T-M2-Fiber','I-Sand Defect',
	'I-Glass Scratch','E-AS-Residue','I-Oil Like','T-M1-Fiber',
	'E-ITO1-Hole','P-ITO1-Residue','E-AS-BPADJ','I-Laser Repair',
	'P-AS-NO','T-ITO1-Residue','I-M2-Crack','E-M2-PR Residue',
	'E-M2-Short','T-Brush defect','T-AS-Residue','T-AS-Particle Small',
	'E-M2-Residue','T-M2-Particle','P-M2-Residue','P-AS-Residue',
	'P-M1-Residue','T-M1-Particle','P-M2-Short','T-AS-SiN Hole',
	'P-AS-BPADJ','P-M2-Open','P-M1-Short'
]
#CLASS_NAMES = ['P-M2-Short', 'T-AS-SiN Hole', 'I-M2-Crack', 'E-M2-Residue', 'T-M2-Fiber', 'E-M1-Al #Residue', 'T-AS-Particle Small', 'T-AS-Residue', 'P-AS-Residue', 'E-M2-Short', 'E-AS-BPADJ', 'T-M1-#Particle', 'T-M2-Particle', 'I-Oil Like', 'T-ITO1-Hole', 'I-Glass Scratch', 'I-Sand Defect', 'P-M2-#Residue', 'T-ITO1-Residue', 'E-ITO1-Hole', 'I-Dust', 'I-Others', 'T-Brush defect', 'P-M1-Short', #'T-M1-Fiber', 'P-M2-Open', 'P-ITO1-Residue', 'P-AS-BPADJ', 'E-AS-Residue', 'E-M2-PR Residue', 'I-#Laser Repair', 'P-M1-Residue', 'P-AS-NO'] #Random order
#CLASS_NAMES = [
#	'I-M2-Crack','E-M2-PR Residue',
#	'E-M2-Short','T-Brush defect','T-AS-Residue','T-AS-Particle Small',
#	'E-M2-Residue','T-M2-Particle','P-M2-Residue','P-AS-Residue',
#	'P-M1-Residue','T-M1-Particle','P-M2-Short','T-AS-SiN Hole',
#	'P-AS-BPADJ','P-M2-Open','P-M1-Short','I-Others','E-M1-Al Residue',
#	'I-Dust','T-ITO1-Hole','T-M2-Fiber','I-Sand Defect',
#	'I-Glass Scratch','E-AS-Residue','I-Oil Like','T-M1-Fiber',
#	'E-ITO1-Hole','P-ITO1-Residue','E-AS-BPADJ','I-Laser Repair',
#	'P-AS-NO','T-ITO1-Residue'
#]#Change sqeence
# fmt: on
#CLASS_NAMES = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']
# CLASS_NAMES = [
#    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
# ]
#CLASS_NAMES = ['I-Others','E-M1-Al Residue','T-M2-Fiber','T-M1-Fiber','I-M2-Crack','E-M2-Residue']

def load_voc_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # shuffle(CLASS_NAMES)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )
