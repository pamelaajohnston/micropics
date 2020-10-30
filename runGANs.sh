CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s aphaniz_ori_lab.npz -d aph_p2p_oriToLab > aph_p2p_oritolabout.txt
CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s aphaniz_ori_rd.npz -d aph_p2p_oriToRD > aph_p2p_oritordout.txt
CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s aphaniz_rd_ori.npz -d aph_p2p_rdToOri > aph_p2p_rdtooriout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_lab.npz -d aph_cgan_oriToLab > aph_cgan_oritolabout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_rd.npz -d aph_cgan_oriToRD > aph_cgan_oritordout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_rd_ori.npz -d aph_cgan_rdToOri > aph_cgan_rdtooriout.txt

CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s planktothrix_ori_lab.npz -d pla_p2p_oriToLab > pla_p2p_oritolabout.txt
CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s planktothrix_ori_rd.npz -d pla_p2p_oriToRD > pla_p2p_oritordout.txt
CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s planktothrix_rd_ori.npz -d pla_p2p_rdToOri > pla_p2p_rdtooriout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s planktothrix_ori_lab.npz -d pla_cgan_oriToLab > pla_cgan_oritolabout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s planktothrix_ori_rd.npz -d pla_cgan_oriToRD > pla_cgan_oritordout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s planktothrix_rd_ori.npz -d pla_cgan_rdToOri > pla_cgan_rdtooriout.txt
