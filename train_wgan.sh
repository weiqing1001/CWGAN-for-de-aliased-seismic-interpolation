#!/bin/sh

python wgan.py --mode "train" --input_dir "./tfdir" --max_steps 600000 --batch_size 1 --l1_weight 100.0 --gan_weight 1.0 \
	--output_dir "./output_wgan" --lr 0.0002 --summary_freq 1000 --trace_freq 1000 --save_freq 200000 
