CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_rd_ori1.npz -d aph_cgan2_rdToOri > aph_cgan2_rdtooriout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_lab1.npz -d aph_cgan2_oriToLab > aph_cgan2_oritolabout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_rd1.npz -d aph_cgan2_oriToRD > aph_cgan2_oritordout.txt

CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_rd_ori2.npz -d aph_cgan2_rdToOri2 > aph_cgan2_rdtooriout2.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_lab2.npz -d aph_cgan2_oriToLab2 > aph_cgan2_oritolabout2.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_rd2.npz -d aph_cgan2_oriToRD2 > aph_cgan2_oritordout2.txt
