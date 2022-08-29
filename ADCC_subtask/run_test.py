import io
import os
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from transformers import (AutoConfig, 
                          AutoModelForTokenClassification,
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score

from dataset  import TCDataset
from test     import TCTest
import utils
import csv


parser = argparse.ArgumentParser()

parser.add_argument('--max_length',
                    default=None,
                    type=int,
                    help='(int) Pad or truncate text sequences to a specific length. If `None` it will use maximum sequence of word piece tokens allowed by model.' )

parser.add_argument('--model_name_or_path',
                    default="-large",
                    type=str,
                    help='(str) Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk.')

parser.add_argument('--test_path',
                    default='./PDTB/implicit_pdtb3_xval/fold_1/test.tsv',
                    type=str,
                    help='(str) Valid tsv/csv path.')

parser.add_argument('--tokenizer',
                    default=None,
                    type=str,
                    help='Name of tokenizer')

parser.add_argument('--config',
                    default=None,
                    type=str,
                    help='Name of config')

args = parser.parse_args()


set_seed(2022)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
labels_to_ids = {
    'O': 0, 'B-Arg1': 1, 'I-Arg1': 2, 'B-Arg2': 3, 'I-Arg2': 4, 'B-Conn': 5, 'I-Conn': 6
    }
label_reversed_dict = {v:k for k, v in labels_to_ids.items()}

n_labels = len(labels_to_ids)

print('Loading configuration...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.config, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.tokenizer)

print('Loading model...')
model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=model_config)

model.to(device)
print('Model loaded to `%s`'%device)

print('Dealing with Train...')
test_dataset = TCDataset(
    path = args.test_path, 
    use_tokenizer = tokenizer, 
    sentences_col = 'sent_tokens',
    labels_to_ids =labels_to_ids,
    labels_col = 'sent_tags',
    max_sequence_len = args.max_length,
    specify_model_type = 'roberta'
    )
print('Created `test_dataset` with %d examples!'%len(test_dataset))

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))
#python inference.py --model_name_or_path "checkpoint/ADCCall+EntRel_2e-06_bert-base-uncased_fold_1.pt" --tokenizer "bert-base-uncased" --config "bert-base-uncased" --max_length 256 --test_path "../PDTB_dataset/EDiTS_datasets/all+EntRel/fold_1/test.json"
print('test on batches...')
#list of lists
test_labels, test_predictions = TCTest.test(model, test_dataloader, device, label_reversed_dict)

"""""""GIVE STATS"""""""
print('giving stats...')
accuracy, F1, recall, precision = utils.partial_scores(test_labels, test_predictions)
arg_accuracy, arg_F1, arg_recall, arg_precision, conn_accuracy, conn_F1, conn_recall, conn_precision = utils.partial_scores_separated(test_labels, test_predictions)

"""""""SAVE RESULTS"""""""
print('save results...')
score_file_name = 'test_result.tsv'
pred_file_name = 'pred_result/' + args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', '') + '_test_pred.tsv'

# write score
if os.path.isfile(score_file_name):
    with open(score_file_name, 'a', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([args.test_path.split('/')[3], args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', ''), args.test_path.split('/')[-2], accuracy, F1, recall, precision, arg_accuracy, arg_F1, arg_recall, arg_precision, conn_accuracy, conn_F1, conn_recall, conn_precision])
    
else:
    with open(score_file_name, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['test_path', 'model', 'fold', 'accuracy', 'f1', 'recall', 'precision', 'arg_accuracy', 'arg_f1', 'arg_recall', 'arg_precision', 'conn_accuracy', 'conn_f1', 'conn_recall', 'conn_precision'])
        w.writerow([args.test_path.split('/')[3], args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', ''), args.test_path.split('/')[-2], accuracy, F1, recall, precision, arg_accuracy, arg_F1, arg_recall, arg_precision, conn_accuracy, conn_F1, conn_recall, conn_precision])

# write result
with open(pred_file_name, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['sent_tokens', 'sent_tag', 'prediction'])
    for sent_tokens, sent_tag, prediction in zip(test_dataset.sentences, test_labels, test_predictions):
        w.writerow([sent_tokens, sent_tag, prediction])