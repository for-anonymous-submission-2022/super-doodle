import os
import time

idx_list = [1,2,4,5,6,7,8,9,11,12]
print("Training for EDRC")
for idx in idx_list:
    start = time.time()
    os.system(f'nohup python -u run.py '
                '--epochs 10 '
                '--model_name_or_path "roberta-base" '
                '--batch_size 8 '
                '--max_length 256 '
                '--learning_rate 2e-6 '
                f'--train_path "../PDTB_dataset/EDiTS_datasets/exp-L2-14way/fold_{idx}/train.json" '
                f'--valid_path "../PDTB_dataset/EDiTS_datasets/exp-L2-14way/fold_{idx}/dev.json" '
                f'> nohup/roberta-base__exp-L2-14way_fold_{idx}'
    )
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")