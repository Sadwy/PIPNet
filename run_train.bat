@REM ######################################################################################
@REM # supervised learning

@REM # 300W, resnet18
python lib/train.py experiments/data_300W/pip_32_16_60_r18_l2_l1_10_1_nb10.py
@REM # 300W, resnet101
@REM python lib/train.py experiments/data_300W/pip_32_16_60_r101_l2_l1_10_1_nb10.py

@REM # COFW, resnet18
@REM python lib/train.py experiments/COFW/pip_32_16_60_r18_l2_l1_10_1_nb10.py
@REM # COFW, resnet101
@REM python lib/train.py experiments/COFW/pip_32_16_60_r101_l2_l1_10_1_nb10.py

@REM # WFLW, resnet18
@REM python lib/train.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py
@REM # WFLW, resnet101
@REM python lib/train.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py

@REM # AFLW, resnet18
@REM python lib/train.py experiments/AFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py
@REM # AFLW, resnet101
@REM python lib/train.py experiments/AFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py

@REM ######################################################################################
@REM # GSSL

@REM # 300W + COFW_68 (unlabeled) + WFLW_68 (unlabeled), resnet18, with curriculum
@REM python lib/train_gssl.py experiments/data_300W_COFW_WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py

@REM # 300W + CelebA (unlabeled), resnet18, with curriculum
@REM nohup python lib/train_gssl.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py &


