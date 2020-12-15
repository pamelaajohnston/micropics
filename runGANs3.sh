CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s aphaniz_ed_ori.npz -d aph_p2p_edToOri > aph_p2p_edtooriout.txt
CUDA_VISIBLE_DEVICES=7 python pix2pixKeras.py -s aphaniz_ori_ed.npz -d aph_p2p_oriToED > aph_p2p_oritoedout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ori_ed.npz -d aph_cgan_oriToED > aph_cgan_oritoedout.txt
CUDA_VISIBLE_DEVICES=7 python cycleGANKeras.py -s aphaniz_ed_ori.npz -d aph_cgan_edToOri > aph_cgan_edtooriout.txt
