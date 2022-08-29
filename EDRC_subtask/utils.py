from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

def list_match(labels:list, predictions:list, labels_pair:list = None): 
    match = 0
    for idx in range(0,len(labels)):
        if labels[idx] == predictions[idx]:
            match += 1
        else:
            # If there are multiple labels, treat the second label(label2) as answer  
            if labels_pair != None:
                if labels_pair[idx] == predictions[idx]:
                    match += 1
    return match

def partial_scoring(labels:list, predictions:list, labels_pair:list): 
    adjusted_labels = []
    for idx, prediction in enumerate(predictions):
        if prediction == labels[idx]:
            adjusted_label = labels[idx]
        elif prediction == labels_pair[idx]:
            adjusted_label = labels_pair[idx]
        else:
            adjusted_label = labels[idx]
        adjusted_labels.append(adjusted_label)
    return accuracy_score(adjusted_labels, predictions), f1_score(adjusted_labels, predictions, average='weighted'), precision_score(adjusted_labels, predictions, average='weighted'), recall_score(adjusted_labels, predictions, average='weighted')

# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

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