CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_rd_ori_sub1.npz -d aph_cgan2_rdToOri > aph_cgan2_rdtooriout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_lab_sub1.npz -d aph_cgan2_oriToLab > aph_cgan2_oritolabout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_rd_sub1.npz -d aph_cgan2_oriToRD > aph_cgan2_oritordout.txt
