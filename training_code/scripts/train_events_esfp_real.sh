python3 ../train.py --gpu 3 \
                 --dataset "realevents"\
                 --dataroot real_dataset/ \
                 --pretrained training_code/results/events_8bin_cvgri/ \
                 --netinput events_8_bins_cvgri \
                 --batch_size 4 \
                 --exp_name "events_cvgri_Real_pretrained"