#Base 17
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/base_2.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005
# 17 + 5
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/2_p_2.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005
# 17 + 10 _ ft
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_2_p_2.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005

sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/4_p_2.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005
sleep 10
python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_4_p_2.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005

#sleep 10
#python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/19_p_1.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005
#sleep 10
#python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_19_p_1.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005

#sleep 10
#python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/22_p_4.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005

#sleep 10
#python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/ft_22_p_4.yaml SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005


#sleep 10
#python tools/visualize_data.py --source annotation --config-file ./configs/PascalVOC-Detection/iOD/ft_15_p_5.yaml --opts ./output/15_p_5_ft/model_final.pth --output-dir ./result/
