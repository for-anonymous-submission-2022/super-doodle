import io
import os
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )

from dataset import EDRC_Dataset
from test import EDRC_Test

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


# parser.add_argument('--learning_rate',
#                     default=5e-6,
#                     type=float,
#                     help='learning rate')

args = parser.parse_args()


# Set seed for reproducibility,
set_seed(2022)

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids =\
    {
        'Expansion.Conjunction'       :0   ,
        'Comparison.Concession'       :1   ,
        'Contingency.Cause'           :2   ,
        'Temporal.Asynchronous'       :3   ,
        'Temporal.Synchronous'        :4   ,
        'Contingency.Condition'       :5   ,
        'Comparison.Contrast'         :6   ,
        'Expansion.Manner'            :7   ,  
        'Contingency.Purpose'         :8   ,
        'Expansion.Instantiation'     :9   ,
        'Expansion.Level-of-detail'   :10  ,
        'Expansion.Substitution'      :11  ,
        'Expansion.Disjunction'       :12  ,
        'Contingency.Neg-condition'   :13  ,
        'Comparison.Similarity'       :14  ,
        'Contingency.Cond+SpeechAct'  :15  ,
        'Contingency.Cause+Belief'    :16  ,
        'Expansion.Exception'         :17  ,
        'Expansion.Equivalence'       :18  ,
        'Comparison.Conc+SpeechAct'   :19  ,
        'Contingency.Cause+SpeechAct' :20  ,
        'Contingency.Negative-cause'  :21  , 
        'NotCon'                      :22  ,
        'NotMat'                      :23  , 
        'EntRel'                      :24  ,
        'None'                        :25
    }

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

# prediction to text label
label_reversed_dict = {v:k for k, v in labels_ids.items()}

# Get model configuration.
print('Loading configuration...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.config, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.tokenizer)

# Get the actual model. (best ckpt)
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)

# Create pytorch dataset.
test_dataset =\
 EDRC_Dataset(path=args.test_path, 
                    use_tokenizer=tokenizer, 
                    sent_col = 'sent_tokens',
                    sent_tags_col = 'sent_tags',
                    labels_ids =labels_ids,
                    labels_col ='sense1',
                    labels_col_pair ='sense2',
                    max_sequence_len = args.max_length)
print('Created `test_dataset` with %d examples!'%len(test_dataset))

# Move pytorch dataset into dataloader.
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))

# Get prediction form model on test data. 
print('test on batches...')
test_labels, test_pairs, test_predictions = EDRC_Test.test(model, test_dataloader, device, True, label_reversed_dict)

"""""""GIVE STATS"""""""
print('giving stats...')
accuracy, F1, precision, recall = utils.partial_scoring(test_labels, test_predictions, test_pairs)

"""""""SAVE RESULTS"""""""
print('save results...')
# loss, acc, prediction result save
score_file_name = 'test_result.tsv'
pred_file_name = 'pred_result/' + args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', '') + '_test_pred.tsv'

# write score
if os.path.isfile(score_file_name):
    with open(score_file_name, 'a', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([args.test_path.split('/')[3], args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', ''), args.test_path.split('/')[-2], accuracy, F1, recall, precision])
    
else:
    with open(score_file_name, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['test_path', 'model', 'fold', 'accuracy', 'f1', 'recall', 'precision'])
        w.writerow([args.test_path.split('/')[3], args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', ''), args.test_path.split('/')[-2], accuracy, F1, recall, precision])

# write result
with open(pred_file_name, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['input', 'label', 'label_pair', 'prediction'])
    for input_text, label, label_pair, prediction in zip(test_dataset.texts, test_labels, test_pairs, test_predictions):
        w.writerow([input_text, label, label_pair, prediction])