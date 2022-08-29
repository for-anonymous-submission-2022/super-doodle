"""
Algorithm from <Implicit Discourse Relation Classification: We Need to Talk about Evaluation> 
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics
@najoungkim | https://github.com/najoungkim/pdtb3
"""

import os
import json

def tab_delimited(list_to_write):
    """Format list of elements in to a tab-delimited line"""
    list_to_write = [str(x) for x in list_to_write]
    return '\t'.join(list_to_write) + '\n'

def write_to_file(json_to_write_d, write_path):
    """Save splits into file"""
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    for split, json_to_write_c in json_to_write_d.items():
        write_fname = '{}.json'.format(split)
        with open(os.path.join(write_path, write_fname), 'w') as f:
            json.dump(json_to_write_c, f)