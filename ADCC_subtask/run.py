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

from dataset  import TCDataset
from train    import TCTrainer
from validate import TCValidation
import utils
import csv

# early stopping
from utils import EarlyStopping


parser = argparse.ArgumentParser()

parser.add_argument('--model_name_or_path',
                    default="bert-large-cased",
                    type=str,
                    help='(str) Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk.')
parser.add_argument('--epochs',
                    default=5,
                    type=int,
                    help='(int) Number of training epochs')
parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length batch of 10 works without cuda memory issues. For small sequence length can try batch of 32 or higher.')
parser.add_argument('--max_length',
                    default=None,
                    type=int,
                    help='(int) Pad or truncate text sequences to a specific length. If `None` it will use maximum sequence of word piece tokens allowed by model.' )
parser.add_argument('--learning_rate',
                    default=2e-5,
                    type=float,
                    help='learning rate')

parser.add_argument('--train_path',
                    default='../PDTB_dataset/pdtb3_xval/fold_1/train.json',
                    type=str,
                    help='(str) Train json/tsv/csv path.')
parser.add_argument('--valid_path',
                    default='../PDTB_dataset/pdtb3_xval/fold_1/dev.json',
                    type=str,
                    help='(str) Valid json/tsv/csv path.')
parser.add_argument('--sentences_col',
                    default='sent_tokens',
                    type=str,
                    help='(str) Valid tsv/csv path.')
parser.add_argument('--labels_col',
                    default='sent_tags',
                    type=str,
                    help='(str) Valid tsv/csv path.')


args = parser.parse_args()

# Set seed for reproducibility,
set_seed(2022)

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_to_ids = {
    'O': 0, 'B-Arg1': 1, 'I-Arg1': 2, 'B-Arg2': 3, 'I-Arg2': 4, 'B-Conn': 5, 'I-Conn': 6
    }
ids_to_labels = {k: v for v, k in labels_to_ids.items()}
print(labels_to_ids)
print(ids_to_labels)

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_to_ids)

# Get model configuration.
print('Loading configuration...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)

# Get the actual model.
print('Loading model...')
model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)

print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = TCDataset(
    path = args.train_path, 
    use_tokenizer = tokenizer, 
    sentences_col = 'sent_tokens',
    labels_to_ids =labels_to_ids,
    labels_col = 'sent_tags',
    max_sequence_len = args.max_length,
    specify_model_type = 'roberta'
    )
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

# Create pytorch dataset.
valid_dataset = TCDataset(
    path = args.valid_path, 
    use_tokenizer = tokenizer, 
    sentences_col = 'sent_tokens',
    labels_to_ids = labels_to_ids,
    labels_col = 'sent_tags',
    max_sequence_len = args.max_length,
    specify_model_type = 'roberta'
    )
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = args.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives 
# us the number of batches.
total_steps = len(train_dataloader) * args.epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}
all_overlap_score = {'train_overlap_score':[], 'val_overlap_score':[]}


# early_stopping when val_loss did not improve for 1 ecpoh.
patience = 1
# checkpoint_name:  ./checkpoint/model_fold_checkpoint.pt
now = datetime.now()

checkpoint_name = 'checkpoint/' + 'ADCC' + str(args.train_path.split('/')[3]) +'_' +  str(args.learning_rate) + '_' +  args.model_name_or_path.split('/')[-1] + '_' + args.train_path.split('/')[-2] + '.pt'  

if not os.path.exists('checkpoint/'):
    os.mkdir('checkpoint/')

early_stopping = EarlyStopping(patience = patience, verbose = True, path=checkpoint_name)

prev_model = None
# Loop through each epoch.
for epoch in range(args.epochs):
    print(f'Epoch: {epoch+1}/{args.epochs}')
    print('Training on batches...')
    # Perform one full pass over the training set.
    model, train_labels, train_predictions, train_loss = TCTrainer.train(model, train_dataloader, optimizer, scheduler, device)
    #train_acc = accuracy_score(train_labels, train_predictions)
    train_overlap_score = utils.token_overlap(train_labels, train_predictions)

    # Get prediction form model on validation data. 
    print('Validation on batches...')
    valid_labels, valid_predictions, val_loss = TCValidation.validation(model, valid_dataloader, device)
    #val_acc = accuracy_score(valid_labels, valid_predictions)
    val_overlap_score = utils.token_overlap(valid_labels, valid_predictions)

    # Print loss and accuracy values to see how training evolves.
    print("  train_loss: %.5f - val_loss: %.5f - train_overlap_score: %.5f - val_overlap_score: %.5f "%(train_loss, val_loss, train_overlap_score, val_overlap_score))

    # Store the loss value for plotting the learning curve.
    all_loss['train_loss'].append(train_loss)
    all_loss['val_loss'].append(val_loss)
    #all_acc['train_acc'].append(train_acc)
    #all_acc['val_acc'].append(val_acc)
    all_overlap_score['train_overlap_score'].append(train_overlap_score)
    all_overlap_score['val_overlap_score'].append(val_overlap_score)

    # loss, acc result save
    if os.path.isfile('result.tsv'):
        with open('result.tsv', 'a', newline='') as f:
            a = csv.writer(f, delimiter='\t')
            a.writerow([args.train_path.split('/')[3], args.model_name_or_path, args.train_path.split('/')[-2], epoch+1, train_loss, val_loss, train_overlap_score, val_overlap_score])
    else:
        with open('result.tsv', 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['train_path', 'model', 'fold', 'epoch', 'train_loss', 'val_loss',  'train_overlap_score', 'val_overlap_score'])
            w.writerow([args.train_path.split('/')[3], args.model_name_or_path, args.train_path.split('/')[-2], epoch+1, train_loss, val_loss, train_overlap_score, val_overlap_score])

            
    # Early stopping
    # 감소했을 경우 현재 모델을 checkpoint로 만듦.
    last = False
    if epoch == args.epochs - 1:
        last = True
    current_val_loss = val_loss
    early_stopping(current_val_loss, prev_model, last)
    prev_model = model

    # best model이 저장되어있는 last checkpoint를 로드한다.
    # model.load_state_dict(torch.load('checkpoint.pt'))
    if early_stopping.early_stop:
        print("----- Early stopping at epoch {} -----".format(epoch))
        print(" ----- ", checkpoint_name, " ----- ")
        break

print(all_loss)
print(all_acc)

# Plot loss curves.
# plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
# Plot accuracy curves.
# plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])


