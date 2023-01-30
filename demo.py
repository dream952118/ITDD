import os
import cv2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Get image
#im = cv2.imread("/home/joseph/PycharmProjects/detectron2/datasets/VOC2007/JPEGImages/000112.jpg")


root = './datasets/AUOcorrectVOC2007/JPEGImages/T-M2-Fiber/'
for i in os.listdir(root):
    pim = root+i
    im = cv2.imread(pim)
    # Get the configuration ready
    cfg = get_cfg()
    cfg.merge_from_file("./configs/PascalVOC-Detection/iOD/ft_22_p_4.yaml")
    cfg.MODEL.WEIGHTS = "./AUOtest/auo33ITDD06output/22_p_4_ft/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)

    v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    img = v.get_image()[:, :, ::-1]
    cv2.imwrite("./itddauoresult/T-M2-Fiber/"+i, img)
