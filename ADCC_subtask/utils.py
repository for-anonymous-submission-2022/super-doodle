from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score

"""token_overlap"""
def token_overlap(labels:list, predictions:list): 
    overlap_scores = []
    for label_idx, label in enumerate(labels):
        match = 0
        total = 0
        for token_idx, token in enumerate(label):
            total += 1
            if token == predictions[label_idx][token_idx]:
                match += 1
        if total != 0:
            overlap_scores.append(match/total)
    return sum(overlap_scores)/len(overlap_scores)

"""token_overlap"""
def partial_scores(labels:list, predictions:list): 
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    for label_idx, label in enumerate(labels):
        match = 0
        total = 0
        prediction = predictions[label_idx]
        for token_idx, token in enumerate(label):
            total += 1
            if token == prediction[token_idx]:
                match += 1
        if match/total > 0.7:
            prediction = label
        accuracy_scores.append(accuracy_score([label], [prediction]))
        f1_scores.append(f1_score([label], [prediction]))
        recall_scores.append(recall_score([label], [prediction]))
        precision_scores.append(precision_score([label], [prediction]))
    return sum(accuracy_scores)/len(accuracy_scores), sum(f1_scores)/len(f1_scores), sum(recall_scores)/len(recall_scores), sum(precision_scores)/len(precision_scores)

"""token_overlap"""
def partial_scores_separated(labels:list, predictions:list): 
    acc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    conn_acc_scores = []
    conn_f1_scores = []
    conn_recall_scores = []
    conn_precision_scores = []

    for label_idx, label in enumerate(labels):
        prediction = predictions[label_idx]
        arg_labels = []
        arg_preds = []
        arg_match = 0
        arg_total = 0
        conn_labels = []
        conn_preds = []
        conn_match = 0
        conn_total = 0
        for token_idx, token in enumerate(label):
            if 'Conn' in token:
                conn_total += 1
                conn_labels.append(token)
                conn_preds.append(prediction[token_idx])
                if token == prediction[token_idx]:
                    conn_match += 1
            elif 'Arg' in token:
                arg_total += 1
                arg_labels.append(token)
                arg_preds.append(prediction[token_idx])
                if token == prediction[token_idx]:
                    arg_match += 1
        if arg_match/arg_total > 0.7:
            arg_match = arg_total

        if arg_total != 0:
            acc_scores.append(arg_match/arg_total)
        if conn_total != 0:
            conn_acc_scores.append(conn_match/conn_total)

        f1_scores.append(f1_score([arg_labels], [arg_preds]))
        recall_scores.append(recall_score([arg_labels], [arg_preds]))
        precision_scores.append(precision_score([arg_labels], [arg_preds]))
        if len(conn_labels) > 0 :
            conn_f1_scores.append(f1_score([conn_labels], [conn_preds]))
            conn_recall_scores.append(recall_score([conn_labels], [conn_preds]))
            conn_precision_scores.append(precision_score([conn_labels], [conn_preds]))
    return sum(acc_scores)/len(acc_scores), sum(f1_scores)/len(f1_scores), sum(recall_scores)/len(recall_scores), sum(precision_scores)/len(precision_scores), sum(conn_acc_scores)/len(conn_acc_scores), sum(conn_f1_scores)/len(conn_f1_scores), sum(conn_recall_scores)/len(conn_recall_scores), sum(conn_precision_scores)/len(conn_precision_scores)


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, prev_model, last):

        score = -val_loss
        if prev_model != None:
            if self.best_score is None:
                self.best_score = score
    #             self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta or last == True: 
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.save_checkpoint(val_loss, prev_model)
                    self.early_stop = True
            else:
                self.best_score = score
    #             self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss