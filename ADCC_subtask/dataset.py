import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np


def select_sep(path: str):
    # Get file from path.
    if path[-3:] == "tsv":
        sep = "\t"
    elif path[-3:] == "csv":
        sep = ","
    elif path[-4:] == "json":
        sep = False
    return sep



class TCDataset(Dataset):
    def __init__(self, path: str, use_tokenizer: object, sentences_col: str, labels_to_ids: dict, labels_col: str, specify_model_type: any = None, max_sequence_len: int = None):
        # Check path.
        if not os.path.isfile(path):
            raise ValueError('Invalid `path` variable! Needs to be a file')

        # Check max sequence length. If not defined, use default.
        max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.sentences = []
        self.labels = []
        
        # Read file
        sep = select_sep(path)
        if sep != False:
            data = pd.read_csv(path, sep=sep, on_bad_lines='warn')
        elif sep == False:
            data = pd.read_json(path, orient='records')
        else:
            print(f"data format at {path} is not supported")

        # Go through content.
        print(f'Reading {path}...')
        data_dict = data.to_dict('records')
        for idx, row in tqdm(enumerate(data_dict)):
            try:
                content = row[sentences_col].strip().split()
                self.sentences.append(content) # Save content.
                tags = row[labels_col].strip().split(",")
                label_id = [labels_to_ids[tag] for tag in tags]
                self.labels.append(label_id) # Save encode labels.
            except AttributeError:
                print(f"SKIPPED {row}, must be str, type: {type(row[sentences_col])} - {type(row[labels_col])} \n {row[sentences_col]} - {row[labels_col]}")

        # Number of instances.
        self.n_instances = len(self.labels)

        # Use tokenizer on sentences. This can take a while.
        print('Using tokenizer on all sentences. This can take a while...')
        self.inputs = use_tokenizer(
            self.sentences, is_split_into_words = True, add_special_tokens = True, truncation = True, padding = 'max_length', return_tensors = 'pt', max_length = max_sequence_len
            )

        # Get maximum sequence length.
        print('Texts padded or truncated to %d length!' % max_sequence_len)
        print('Adjusting Labels')
        adjusted_labels = []
        for idx, label in tqdm(enumerate(self.labels)):
            word_ids = self.inputs.word_ids(batch_index=idx)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            adjusted_labels.append(label_ids)

        self.inputs.update({'labels':torch.tensor(adjusted_labels)})
        #print(self.inputs)
        print('Finished!\n')

    def __len__(self):
        """When used `len` return the number of instances.

        """
        return self.n_instances


    def __getitem__(self, item):
        """Given an index return an example from the position.
        
        Arguments:

        item (:obj:`int`):
            Index position to pick an example to return.

        Returns:
        :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
        It holddes the statement `model(**Returned Dictionary)`.

        """
        return {key: self.inputs[key][item] for key in self.inputs.keys()}