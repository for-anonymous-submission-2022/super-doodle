import os
import time

print("Training for NDRC")
for idx in range(0,13):
    start = time.time()
    os.system(f'nohup python -u run.py '
                '--epochs 10 '
                '--model_name_or_path "roberta-base" '
                '--batch_size 8 '
                '--max_length 256 '
                '--learning_rate 2e-6 '
                f'--train_path "../PDTB_dataset/EDiTS_datasets/nexp-L2-14way+EntRel/fold_{idx}/train.json" '
                f'--valid_path "../PDTB_dataset/EDiTS_datasets/nexp-L2-14way+EntRel/fold_{idx}/dev.json" '
                f'> nohup/roberta-base__nexp-L2-14way+EntRel_fold_{idx}'
    )
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")