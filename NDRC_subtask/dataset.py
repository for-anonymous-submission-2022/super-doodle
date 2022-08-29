import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from utils import multiple_replace


def select_sep(path: str):
    # Get file from path.
    if path[-3:] == "tsv":
        sep = "\t"
    elif path[-3:] == "csv":
        sep = ","
    elif path[-4:] == "json":
        sep = False
    return sep

def make_NDRC_input(sent_tokens: list, sent_tags: list):
    to_delete = []
    for idx, sent_tag in enumerate(sent_tags):
        if sent_tag == "O":
            to_delete.append(idx)
    for idx in sorted(to_delete, reverse=True):
        del sent_tokens[idx]
    return sent_tokens

class NDRC_Dataset(Dataset):

    def __init__(self, path: str, use_tokenizer: object, sent_col: str, sent_tags_col: str, labels_ids: dict, labels_col: str, labels_col_pair: str = None, max_sequence_len: int = None):
        # Check path.
        if not os.path.isfile(path):
            raise ValueError('Invalid `path` variable! Needs to be a file')

        # Check max sequence length. If not defined, use default.
        max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.texts = []
        self.texts_pair = []
        self.labels = []
        self.labels_pair = []
        
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

        for row in tqdm(data_dict):

            sent_tokens = row[sent_col].strip().split()
            sent_tags = row[sent_tags_col].strip().split(",")
            
            content = make_NDRC_input(sent_tokens, sent_tags)
            content = ' '.join(content)
            self.texts.append(content)

            label_id = labels_ids[row[labels_col]]
            self.labels.append(label_id) # Save encode labels.
            if labels_col_pair != None:
                if row[labels_col_pair] == None:
                    label_id_pair = labels_ids['None']
                else:
                    label_id_pair = labels_ids[row[labels_col_pair]]

                self.labels_pair.append(label_id_pair)

        # Number of instances.
        self.n_instances = len(self.labels)

        # Use tokenizer on texts. This can take a while.
        print('Using tokenizer on all texts. This can take a while...')
        self.inputs = use_tokenizer(self.texts, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length=max_sequence_len)

        # Get maximum sequence length.
        print('Texts padded or truncated to %d length!' % max_sequence_len)

        # Add labels.
        self.inputs.update({'labels':torch.tensor(self.labels)})
        if labels_col_pair != None:
            self.inputs.update({'labels_pair':torch.tensor(self.labels_pair)})
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