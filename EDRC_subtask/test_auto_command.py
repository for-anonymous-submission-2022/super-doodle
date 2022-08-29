import os
import time


print("Testing for EDSC")
for idx in range(1,13):
    start = time.time()
    os.system(f'nohup python -u run_test.py '
                f'--model_name_or_path "checkpoint/EDRCexp-L2-14way_2e-06_bert-base-uncased_fold_{idx}.pt" '
                '--tokenizer "bert-base-uncased" '
                '--config "bert-base-uncased" '
                '--max_length 256 '
                f'--test_path "../PDTB_dataset/EDiTS_datasets/exp-L2-14way/fold_{idx}/test.json"'
    )
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

print("Testing for EDSC")
for idx in range(1,13):
    start = time.time()
    os.system(f'nohup python -u run_test.py '
                f'--model_name_or_path "checkpoint/EDRCexp-L2-14way_2e-06_roberta-base_fold_{idx}.pt" '
                '--tokenizer "roberta-base" '
                '--config "roberta-base" '
                '--max_length 256 '
                f'--test_path "../PDTB_dataset/EDiTS_datasets/exp-L2-14way/fold_{idx}/test.json"'
    )
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")