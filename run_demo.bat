@REM image
@REM python lib/demo.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py images/1.jpg
@REM python lib/demo.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py images/2.jpg

@REM video
@REM python lib/demo_video.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py videos/002.avi
@REM python lib/demo_video.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py videos/007.avi

@REM camera
python lib/demo_video0.2.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera
python lib/demo_video_test.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera
python lib/demo_video_test_stable.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera

