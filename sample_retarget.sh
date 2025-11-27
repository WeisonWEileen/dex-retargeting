#!/bin/bash

WANDB_API_KEY="70b35cc989ebf8652e52516c433f9faa444d21d2"
CUDA_VISIBLE_DEVICES="0" \
/datasets/v2p/current/pw-workspace/conda/dex-retarget/bin/python \
example/position_retargeting/visualize_hand_object.py \
--dexycb-dir=/home/ghr/panwei/pw-workspace/dex_latent/data \
--robots shadow \
--device cuda:0 \
--sample-from-model \
--model-path /home/ghr/panwei/pw-workspace/dex_latent/output/20251113_103740_beta0.1_seed42/20251113_104312_beta0.01_seed42/checkpoints/epoch_60.pth
