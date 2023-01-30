#Base 17
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/base_17.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
# 17 + 5
#sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/17_p_5.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
# 17 + 10 _ ft
#sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_17_p_5.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005


sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/22_p_4.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
# 22 + 4 _ ft
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_22_p_4.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005

sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/26_p_4.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
# 26 + 4 _ ft
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_26_p_4.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005

sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/30_p_4.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
# 30 + 4 _ ft
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_30_p_4.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005

sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/19_p_1.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
# 30 + 4 _ ft
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_19_p_1.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005

sleep 10
python tools/visualize_data.py --source annotation --config-file ./configs/PascalVOC-Detection/iOD/ft_19_p_1.yaml --opts ./output/19_p_1_ft/model_final.pth --output-dir ./result/