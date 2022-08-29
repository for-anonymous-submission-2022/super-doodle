"""
PDTB 3.0 | Preprocessing Code | Layman's Parser

--- Credits ---
x-Fold Validation | Splitting Algorithm Adopted from 
Publication: Implicit Discourse Relation Classification: We Need to Talk about Evaluation (Kim et al., ACL 2020)
@najoungkim | https://github.com/najoungkim/pdtb3
"""

import re
import os
import shutil
import random
import argparse
import itertools
import copy
import json

import tqdm
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize')
import spacy
nlp_s = spacy.load("en_core_web_sm")

import pdtb3_utils
import pdtb3_sense

def retrieve_basics(which_dataset):
    global NON_EXPLICIT_RELATIONS
    global EXPLICIT_RELATIONS
    global ALL_RELATIONS
    global SELECTED_SENSES_PDTB3
    global UNSELECTED_SENSES_PDTB3

    NON_EXPLICIT_RELATIONS =\
        pdtb3_sense.defined_sets("NON_EXPLICIT_RELATIONS")
    EXPLICIT_RELATIONS =\
        pdtb3_sense.defined_sets("EXPLICIT_RELATIONS")
    ALL_RELATIONS =\
        pdtb3_sense.defined_sets("ALL_RELATIONS")
    SELECTED_SENSES_PDTB3 =\
        pdtb3_sense.defined_sets("considered_"+which_dataset)
    UNSELECTED_SENSES_PDTB3 =\
        pdtb3_sense.defined_sets("notconsidered_"+which_dataset)




def start_process_raw(data_path, write_path, select_relations):
    total_intra_sentential_match = 0
    total_inter_sentential_match = 0
    total_fully_tagged_sentences = 0
    total_unmatched_sentences    = 0
    total_number_of_sentences    = 0

    annotation_path = os.path.join(data_path, 'gold/')
    text_path = os.path.join(data_path, 'raw/')
    
    for dirname in os.listdir(annotation_path):
        json_to_write = []
        annotation_dir = os.path.join(annotation_path, dirname)
        text_dir = os.path.join(text_path, dirname)

        for filename in os.listdir(annotation_dir):
            with open(os.path.join(annotation_dir, filename), encoding='latin1') as f:
                annotation_data = f.readlines()
            with open(os.path.join(text_dir, filename), encoding='latin1') as f:
                text_data = f.read()

            doc = nlp(text_data)
            text_data_by_sent = [sentence.text for sentence in doc.sentences]

            results, file_stats = process_file(annotation_data, text_data, text_data_by_sent, dirname, filename, select_relations)

            total_intra_sentential_match += file_stats[0] 
            total_inter_sentential_match += file_stats[1] 
            total_fully_tagged_sentences += file_stats[2] 
            total_unmatched_sentences    += file_stats[3] 
            total_number_of_sentences    += len(text_data_by_sent) 

            for result in results:
                #there must be 14 columns
                assert len(result) == 14, f"length is wrong, {result}"
                
            json_to_write.extend(results)

        with open(f'{write_path}/{dirname}.tsv', 'w') as f:
            json.dump(json_to_write, f)
            print(f'Wrote Section {dirname}')

    #PDTB-3 counts: 22640,19912,42434,14277,51292
    print(f"total_intra_sentential_match : {total_intra_sentential_match} \n"
          f"total_inter_sentential_match : {total_inter_sentential_match} \n"   
          f"total_fully_tagged_sentences : {total_fully_tagged_sentences} \n"
          f"total_unmatched_sentences    : {total_unmatched_sentences   } \n"
          f"total_number_of_sentences    : {total_number_of_sentences   } \n")





def process_file(annotation_data, text_data, text_data_by_sent, dirname, filename, select_relations):
    json_to_write = []
    matched_sentences = []
    file_stats = [0,0,0,0]

    ### Find Matching Setence for Each Annotation Line
    for annotation_line in annotation_data:
        data_tuple = process_line(annotation_line, text_data, text_data_by_sent, select_relations)
        if data_tuple:
            relation_type, sent_type,          \
            arg1_str, arg2_str,                \
            conn1, conn1_sense1, conn1_sense2, \
            conn2, conn2_sense1, conn2_sense2, \
            sent, sent_tokens, sent_tags,      \
            line_stats                         = data_tuple

            file_stats[0] += 1 if "Intra" in line_stats else 0
            file_stats[1] += 1 if "Inter" in line_stats else 0
            file_stats[2] += 1 if "FullTag" in line_stats else 0
            matched_sentences.append(sent)

            #tokens and tags length must be same
            assert len(sent_tokens) == len(sent_tags), f"length is wrong, {len(sent_tokens)} {len(sent_tags)}"
            if relation_type == "EntRel":
                conn1_sense1 = "EntRel"

            ### Append Matched Instances
            json_to_write.append({
                'section' : dirname, 
                'filename' : filename,
                'relation_type' : relation_type, 
                'sent_type' : sent_type,
                'arg1' : ' '.join(arg1_str), 
                'arg2' : ' '.join(arg2_str),
                'conn1' : conn1, 
                'conn1_sense1' : conn1_sense1, 
                'conn1_sense2' : conn1_sense2,
                'conn2' : conn2, 
                'conn2_sense1' : conn2_sense1, 
                'conn2_sense2' : conn2_sense2,
                'sent_tokens' : ' '.join(sent_tokens), 
                'sent_tags' : ','.join(sent_tags)
                })
    
    ### OPTIONAL: Append Unmatched Instances (those not supported 
    ### by this code or has no discourse relation from get-go)
    for raw_sentence in text_data_by_sent:
        matched_flag = False
        raw_sentence = re.sub('\n', ' ', raw_sentence)
        for matched_sentence in matched_sentences:
            if raw_sentence in matched_sentence:
                # If at least one matching instance, turn flag True
                matched_flag = True
        if matched_flag == False:
            sent_tags = []

            ### Tokenize Sent
            doc = nlp_s(raw_sentence)
            token_list = [token.text for token in doc]
            sent_tokens = [re.sub(' ', '', token) for token in token_list]
            sent_tokens = list(filter(lambda x: x != '', sent_tokens))

            ### Initialize Sent Tags
            for sent_token in sent_tokens:
                sent_tags.append("O")

            file_stats[3] += 1

            #tokens and tags length must be same
            assert len(sent_tokens) == len(sent_tags), f"length is wrong, {len(sent_tokens)} {len(sent_tags)}"

            json_to_write.append({
                'section' : dirname, 
                'filename' : filename,
                'relation_type' : 'NotMat', 
                'sent_type' : 'NotMat',
                'arg1' : '', 
                'arg2' : '',
                'conn1' : '', 
                'conn1_sense1' : 'NotMat', 
                'conn1_sense2' : '',
                'conn2' : '', 
                'conn2_sense1' : '', 
                'conn2_sense2' : '',
                'sent_tokens' : ' '.join(sent_tokens), 
                'sent_tags' : ','.join(sent_tags)
                })
                
    return json_to_write, file_stats




def process_line(annotation_line, text_data, text_data_by_sent, select_relations):
    ### For Stats
    line_stats = []

    ### Prepare Annotation Line
    annotation_line = re.sub('\n', ' ', annotation_line)
    annotation = annotation_line.split('|')

    ### Check Relation
    relation_type = annotation[0]
    if select_relations == "Non-explicit":
        if relation_type not in NON_EXPLICIT_RELATIONS:
            return None
    elif select_relations == "Explicit":
        if relation_type not in EXPLICIT_RELATIONS:
            return None
    elif select_relations == "All":
        if relation_type not in ALL_RELATIONS:
            return None
    else:
        print("select_relations wrong.")
        raise ValueError

    ### Prepare Connective
    conn1 = annotation[7]
    conn1_sense1 = annotation[8]
    conn1_sense2 = annotation[9]
    conn2 = annotation[10]
    conn2_sense1 = annotation[11]
    conn2_sense2 = annotation[12]
    conn_idx = annotation[1].split(';') # may be discontiguous span
    conn_str = [] # maintained separately for tag_sent()

    if relation_type == "Explicit":
        for pairs in conn_idx:
            conn_i, conn_j = pairs.split('..')
            conn = text_data[int(conn_i):int(conn_j)+1]
            conn_str.append(re.sub('\n', ' ', conn))
    else:
        conn_str = ['undef']

    ### Prepare Arguments
    arg1_idx = annotation[14].split(';')
    arg2_idx = annotation[20].split(';')
    arg1_str = []
    arg2_str = []

    for pairs in arg1_idx:
        arg1_i, arg1_j = pairs.split('..')
        arg1 = text_data[int(arg1_i):int(arg1_j)+1]
        arg1_str.append(re.sub('\n', ' ', arg1))

    for pairs in arg2_idx:
        if pairs == '':
            continue
        arg2_i, arg2_j = pairs.split('..')
        arg2 = text_data[int(arg2_i):int(arg2_j)+1]
        arg2_str.append(re.sub('\n', ' ', arg2))

    for idx, raw_sentence in enumerate(text_data_by_sent):
        ### Prepare Sentence
        this_raw_sent = re.sub('\n', ' ', raw_sentence)
        next_raw_sent = re.sub('\n', ' ', text_data_by_sent[idx+1]) if idx+1 != len(text_data_by_sent) else ''

        ### Check if Args and Sent Match
        arg1_second_portion = arg1_str[1] if len(arg1_str) > 1 else arg1_str[0]
        arg2_second_portion = arg2_str[1] if len(arg2_str) > 1 else arg2_str[0]
        if this_raw_sent.find(arg1_str[0]) != -1\
         and this_raw_sent.find(arg2_str[0]) != -1\
          and this_raw_sent.find(arg1_second_portion) != -1\
           and this_raw_sent.find(arg2_second_portion) != -1:
            sent = this_raw_sent
            sent_type = "Intra"
        elif this_raw_sent.find(arg1_str[0]) != -1 and next_raw_sent.find(arg2_second_portion) != -1:
            sent = this_raw_sent + ' ' + next_raw_sent
            sent_type = "Inter"
        elif this_raw_sent.find(arg2_str[0]) != -1 and next_raw_sent.find(arg1_second_portion) != -1:
            sent = this_raw_sent + ' ' + next_raw_sent
            sent_type = "Inter"
        else:
            continue # If annotation line not matched, skip

        ### Tag Matched Sentence
        sent_tokens, sent_tags, FullTag = tag_sent(sent, text_data, arg1_idx, arg2_idx, arg1_str, arg2_str, conn_str)

        line_stats.append(sent_type)
        line_stats.append("FullTag") if FullTag == True else ""

        return (relation_type, sent_type,
                arg1_str, arg2_str,
                conn1, conn1_sense1, conn1_sense2,
                conn2, conn2_sense1, conn2_sense2,
                sent, sent_tokens, sent_tags, 
                line_stats)
        
    return None





def tag_sent(sent, text_data, arg1_idx, arg2_idx, arg1_str, arg2_str, conn_str):
    ### Prepare Tagging
    sent_tokens = []
    arg1_tokens = [] #list of list
    arg2_tokens = [] #list of list
    conn_tokens = [] #list of list
    sent_tags = []
    arg1_tags = [] #list of list
    arg2_tags = [] #list of list
    conn_tags = [] #list of list

    ### Tokenize Sent
    doc = nlp_s(sent)
    token_list = [token.text for token in doc]
    sent_tokens = [re.sub(' ', '', token) for token in token_list]
    sent_tokens = list(filter(lambda x: x != '', sent_tokens))

    ### Initialize Sent Tags
    for sent_token in sent_tokens:
        sent_tags.append("O")

    ### Tokenize Arg1
    for item in arg1_str:
        doc = nlp_s(item)
        token_list = [token.text for token in doc]
        arg1_tokens.append([re.sub(' ', '', token) for token in token_list])

    ### Initialize Arg1 Tags
    for arg1_idx_section, arg1_tokens_section in enumerate(arg1_tokens):
        arg1_tags_section = []
        arg1_tokens[arg1_idx_section] = list(filter(lambda x: x != '', arg1_tokens_section))
        for token in arg1_tokens[arg1_idx_section]:
            arg1_tags_section.append("I-Arg1") 
        arg1_tags.append(arg1_tags_section)
    arg1_tags[0][0] = "B-Arg1"

    ### Tokenize Arg2
    for item in arg2_str:
        doc = nlp_s(item)
        token_list = [token.text for token in doc]
        arg2_tokens.append([re.sub(' ', '', token) for token in token_list])

    ### Initialize Arg2 Tags
    for arg2_idx_section, arg2_tokens_section in enumerate(arg2_tokens):
        arg2_tags_section = []
        arg2_tokens[arg2_idx_section] = list(filter(lambda x: x != '', arg2_tokens_section))
        for token in arg2_tokens[arg2_idx_section]:
            arg2_tags_section.append("I-Arg2") 
        arg2_tags.append(arg2_tags_section)
    arg2_tags[0][0] = "B-Arg2"

    ### Tokenize Conn
    for item in conn_str:
        doc = nlp_s(item)
        token_list = [token.text for token in doc]
        conn_tokens.append([re.sub(' ', '', token) for token in token_list])

    ### Initialize Conn Tags
    for conn_idx_section, conn_tokens_section in enumerate(conn_tokens):
        conn_tags_section = []
        conn_tokens[conn_idx_section] = list(filter(lambda x: x != '', conn_tokens_section))
        for token in conn_tokens[conn_idx_section]:
            if token != 'undef':
                conn_tags_section.append("I-Conn") 
        conn_tags.append(conn_tags_section)
    if conn_tokens[0][0] != 'undef':
        conn_tags[0][0] = "B-Conn"

    ### Token Match (Tag) Sentence Against Arg1, Arg2, and Conn
    sent_tags = token_matcher(arg1_tokens, arg1_tags, sent_tokens, sent_tags)
    sent_tags = token_matcher(arg2_tokens, arg2_tags, sent_tokens, sent_tags)
    sent_tags = token_matcher(conn_tokens, conn_tags, sent_tokens, sent_tags)
    
    ### Check Stats
    FullTag = True
    if sent_tags.count("I-Arg1") + 1 != len(list(itertools.chain(*arg1_tags))):
        FullTag = False
    if sent_tags.count("I-Arg2") + 1 != len(list(itertools.chain(*arg2_tags))):
        FullTag = False
    if list(itertools.chain(*conn_tags)).count("I-Conn") != sent_tags.count("I-Conn")\
    and list(itertools.chain(*conn_tags)).count("B-Conn")!= sent_tags.count("B-Conn"):
        FullTag = False
    
    return sent_tokens, sent_tags, FullTag





def token_matcher(match_tokens, match_tags, sent_tokens, sent_tags):
    for match_idx_section, match_tokens_section in enumerate(match_tokens):
        for sent_idx, sent_token in enumerate(sent_tokens):
            temp_sent_tags = copy.deepcopy(sent_tags)
            if sent_token == match_tokens_section[0] and len(match_tokens_section) <= len(sent_tokens[sent_idx:]) and sent_tags[sent_idx] == "O":
                stopper_flag = False
                for inc in range(len(match_tokens_section)):
                    temp_sent_tags[sent_idx + inc] = match_tags[match_idx_section][inc]
                    if sent_tokens[sent_idx + inc] != match_tokens_section[inc]:
                        stopper_flag = True
                        break
                if stopper_flag == False:
                    sent_tags = temp_sent_tags
                    break
    return sent_tags





def pdtb3_make_community(data_path, write_path, random_sections=False, level='L2'):
    """Creates datasets for community version.

    Note that this method only creates splits based on 14-way classification.
    That is, it will skip low-count labels even if they are relevant
    for the 4-way classification.

    Args:
        data_path: Path containing the PDTB 3.0 data preprocessed into sections.
        write_path: Path to write the splits to.
        random_sections: Whether to create randomized splits or fixed splits.
    """
    dev_sections = ['23']
    test_sections = ['24']
    train_sections = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10','11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21','22']

    means_d = {'train': 0, 'dev': 0, 'test': 0}

    print(f"BUILDING DATASET FOR COMMUNITY VERSION")

    label_d = {}
    all_splits = dev_sections + test_sections + train_sections
    assert len(set(all_splits)) == 25

    split_d = {'train': train_sections, 'dev': dev_sections, 'test': test_sections}
    json_to_write_d = {'train': [], 'dev': [], 'test': []}

    for split, sections in split_d.items():
        for section in sections:
            process_section(data_path, section, split, json_to_write_d, label_d, level)

    for split, lines in json_to_write_d.items():
        means_d[split] += len(lines)-1

    write_path_fold = os.path.join(write_path, f'community')
    pdtb3_utils.write_to_file(json_to_write_d, write_path_fold)

    print(f"labels: {label_d}")

    for split, total in means_d.items():
        print(f'Total: {total}')
        print(f'Mean {split}: {total/len(dev_sections)}')





def pdtb3_make_splits_xval(data_path, write_path, random_sections=False, level='L2'):
    """Creates cross-validation splits.

    Note that this method only creates splits based on 14-way classification.
    That is, it will skip low-count labels even if they are relevant
    for the 4-way classification.

    Args:
        data_path: Path containing the PDTB 3.0 data preprocessed into sections.
        write_path: Path to write the splits to.
        random_sections: Whether to create randomized splits or fixed splits.
    """
    sections = [
        '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10','11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21','22', '23', '24'
        ]
    dev_sections = []
    test_sections = []
    train_sections = []

    means_d = {'train': 0, 'dev': 0, 'test': 0}

    if not random_sections:
        for i in range(0, 25, 2):
            dev_sections.append([sections[i], sections[(i+1)%25]])
            test_sections.append([sections[(i+23)%25], sections[(i+24)%25]])
            train_sections.append([sections[(i+j)%25] for j in range(2, 23)])
    else:
        seed = 111
        random.seed(seed)
        for i in range(0, 13, 1):
            random.seed(seed + i)
            random.shuffle(sections)
            dev_sections.append(sections[0:2])
            test_sections.append(sections[2:4])
            train_sections.append(sections[4:])

    print(f"dev: {dev_sections}")
    print(f"test: {test_sections}")

    for fold_no, (dev, test, train) in enumerate(zip(dev_sections[:-1],
                                                     test_sections[:-1],
                                                     train_sections[:-1])):
        label_d = {}
        all_splits = dev + test + train
        assert len(set(all_splits)) == 25

        split_d = {'train': train, 'dev': dev, 'test': test}
        json_to_write_d = {'train': [], 'dev': [], 'test': []}

        for split, sections in split_d.items():
            for section in sections:
                process_section(data_path, section, split, json_to_write_d, label_d, level)

        for split, lines in json_to_write_d.items():
            means_d[split] += len(lines)-1

        write_path_fold = os.path.join(write_path, f'fold_{fold_no + 1}')
        pdtb3_utils.write_to_file(json_to_write_d, write_path_fold)

    print(f"labels: {label_d}")

    for split, total in means_d.items():
        print(f'Total: {total}')
        print(f'Mean {split}: {total/len(dev_sections[:-1])}')





def process_section(data_path, section, split, json_to_write_d, label_d, level='L2'):
    """Processes a single PDTB section."""
    with open(data_path + '/' + section + '.tsv') as f:
        data = json.load(f)

    for instance in data:
        section = instance['section']     
        file_no = instance['filename']     
        relation_type = instance['relation_type'] 
        sent_type = instance['sent_type']   
        arg1 = instance['arg1']        
        arg2 = instance['arg2']        
        conn1 = instance['conn1']      
        conn1_sense1 = instance['conn1_sense1'] 
        conn1_sense2 = instance['conn1_sense2'] 
        conn2 = instance['conn2']  
        conn2_sense1 = instance['conn2_sense1'] 
        conn2_sense2 = instance['conn2_sense2'] 
        sent_tokens = instance['sent_tokens']   
        sent_tags = instance['sent_tags']     

        sense1 = (conn1_sense1, conn1)
        sense2 = (conn1_sense2, conn1)
        sense3 = (conn2_sense1, conn2)
        sense4 = (conn2_sense2, conn2)

        # Use list instead of set to preserve order
        sense_list = [sense1, sense2, sense3, sense4]
        if level == 'L2':
            formatted_sense_list = format_sense_l2(sense_list)
        else:
            raise ValueError('Level must be L2')

        # No useable senses
        if not formatted_sense_list:
            continue

        if split == 'train':
            for sense, conn, sense_full in formatted_sense_list:
                json_to_write_d[split].append({
                    'split' : split,
                    'section' : section, 
                    'filename' : file_no,
                    'relation_type' : relation_type, 
                    'sent_type' : sent_type,
                    'arg1' : arg1, 
                    'arg2' : arg2,
                    'conn' : conn, 
                    'sense' : sense, 
                    'sense_full' : sense_full,
                    'sent_tokens' : sent_tokens, 
                    'sent_tags' : sent_tags
                    })
                
                label_d[sense] = label_d.get(sense, 0) + 1

                tags = sent_tags.strip().split(",")
                label_d["B-Conn"] = label_d.get("B-Conn", 0) + tags.count("B-Conn")
                label_d["I-Conn"] = label_d.get("I-Conn", 0) + tags.count("I-Conn")
                label_d["B-Arg1"] = label_d.get("B-Arg1", 0) + tags.count("B-Arg1")
                label_d["I-Arg1"] = label_d.get("I-Arg1", 0) + tags.count("I-Arg1")
                label_d["B-Arg2"] = label_d.get("B-Arg2", 0) + tags.count("B-Arg2")
                label_d["I-Arg2"] = label_d.get("I-Arg2", 0) + tags.count("I-Arg2")
                label_d["O"] = label_d.get("O", 0) + tags.count("O")

        else:
            if len(formatted_sense_list) == 1:
                formatted_sense_list.append((None, None, None))
            sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
            senses, conns, senses_full = sense_paired
            json_to_write_d[split].append({
                'split' : split,
                'section' : section, 
                'filename' : file_no,
                'relation_type' : relation_type, 
                'sent_type' : sent_type,
                'arg1' : arg1, 
                'arg2' : arg2,
                'conn1' : conns[0], 
                'sense1' : senses[0], 
                'sense1_full' : senses_full[0],
                'conn2' : conns[1], 
                'sense2' : senses[1], 
                'sense2_full' : senses_full[1],
                'sent_tokens' : sent_tokens, 
                'sent_tags' : sent_tags
                })

            label_d[senses[0]] = label_d.get(senses[0], 0) + 1
            if senses[1] is not None:
                label_d[senses[1]] = label_d.get(senses[1], 0) + 1

            tags = sent_tags.strip().split(",")
            label_d["B-Conn"] = label_d.get("B-Conn", 0) + tags.count("B-Conn")
            label_d["I-Conn"] = label_d.get("I-Conn", 0) + tags.count("I-Conn")
            label_d["B-Arg1"] = label_d.get("B-Arg1", 0) + tags.count("B-Arg1")
            label_d["I-Arg1"] = label_d.get("I-Arg1", 0) + tags.count("I-Arg1")
            label_d["B-Arg2"] = label_d.get("B-Arg2", 0) + tags.count("B-Arg2")
            label_d["I-Arg2"] = label_d.get("I-Arg2", 0) + tags.count("I-Arg2")
            label_d["O"] = label_d.get("O", 0) + tags.count("O")




def format_sense_l2(sense_list):
    formatted_sense_list = []
    for sense_full, conn in sense_list:
        if sense_full is not None:
            sense = '.'.join(sense_full.split('.')[0:2])
            if (sense not in [s for s, c, sf in formatted_sense_list]):
                if sense in SELECTED_SENSES_PDTB3:
                    formatted_sense_list.append((sense, conn, sense_full))
                elif 'NotCon' in SELECTED_SENSES_PDTB3 and sense in UNSELECTED_SENSES_PDTB3:
                    formatted_sense_list.append(('NotCon', conn, sense_full))
    return formatted_sense_list





def pdtb3_make_splits_l1(data_path, write_path):
    """Creates a split for L1 classification using specifications from Ji & Eistenstein (2015)."""
    TRAIN = ['02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    DEV = ['00', '01']
    TEST = ['21', '22']

    label_d = {}

    dev_sections = [DEV]
    test_sections = [TEST]
    train_sections = [TRAIN]

    for (dev, test, train) in zip(dev_sections, test_sections, train_sections):

        split_d = {'train': train, 'dev': dev, 'test': test}
        json_to_write_d = {'train': [], 'dev': [], 'test': []}

        label_d = {}
        for split, sections in split_d.items():
            for section in sections:
                process_section_l1(data_path, section, split, json_to_write_d, label_d)

        # Write to file
        pdtb3_utils.write_to_file(json_to_write_d, write_path)





def process_section_l1(data_path, section, split, json_to_write_d, label_d):
    with open(data_path + '/' + section + '.tsv') as f:
        data = f.readlines()

    for line in data[1:]:
        section, file_no, category, arg1, arg2, \
        conn1, conn1_sense1, conn1_sense2, \
        conn2, conn2_sense1, conn2_sense2 = line.rstrip('\n').split('\t')

        sense1 = (conn1_sense1, conn1)
        sense2 = (conn1_sense2, conn1)
        sense3 = (conn2_sense1, conn2)
        sense4 = (conn2_sense2, conn2)

        sense_list = [sense1, sense2, sense3, sense4]
        formatted_sense_list = []
        for sense_full, conn in sense_list:
            if sense_full is not None:
                sense = sense_full.split('.')[0]
                if sense not in [s for s, c, sf in formatted_sense_list] and sense:
                    formatted_sense_list.append((sense, conn, sense_full))

        # Should be at least one sense
        assert formatted_sense_list
        assert len(formatted_sense_list) <= 2, formatted_sense_list
        if len(formatted_sense_list) == 2:
            assert formatted_sense_list[0] != formatted_sense_list[1]

        if split == 'train':
            for sense, conn, sense_full in formatted_sense_list:
                json_to_write_d[split].append(pdtb3_utils.tab_delimited([split, section, file_no,
                                                     sense, category, arg1,
                                                     arg2, conn, sense_full]))
                label_d[sense] = label_d.get(sense, 0) + 1

        else:
            if len(formatted_sense_list) == 1:
                formatted_sense_list.append((None, None, None))
            sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
            senses, conns, senses_full = sense_paired
            json_to_write_d[split].append(pdtb3_utils.tab_delimited([split, section, file_no,
                                                 senses[0], senses[1], category,
                                                 arg1, arg2, conns[0], senses_full[0],
                                                 conns[1], senses_full[1]]))

            label_d[senses[0]] = label_d.get(senses[0], 0) + 1
            if senses[1] is not None:
                label_d[senses[1]] = label_d.get(senses[1], 0) + 1





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='Path to a directory containing raw and gold PDTB 3.0 files.\
                              Refer to README.md about obtaining this file.')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='Path to output directory \
                              where the preprocessed dataset will be stored.')
    parser.add_argument('--which_dataset', default=None, type=str,
                        help='which dataset to use?')
    parser.add_argument('--split', default="L2_xval", type=str,
                        help='Type of split to create. Should be one of \
                              "L2_community", "L2_xval", or "L1_ji".')
    parser.add_argument('--select_relations', default=None, type=str,
                        help='Non-explicit or Explicit or All?')
    parser.add_argument('--create_sections', default="Yes", type=str,
                        help='Recreate sections/ ? Turn "No" if only \
                              re-creating split (= not preprocessing from the start')
    args = parser.parse_args()
    
    assert args.select_relations in ["Non-explicit", "Explicit", "All"], "select_relations must be Non-explicit, Explicit, All"

    retrieve_basics (args.which_dataset)

    ### Check if sections_data_dir exists
    sections_data_dir = os.path.join(args.data_dir, 'sections/')
    if os.path.exists(sections_data_dir):
        print(f'sections/ present at {sections_data_dir}')
    else:
        print(f'sections/ not present at {sections_data_dir}')

    if args.create_sections == "Yes":
        ### if sections/ exist, delete path
        if os.path.exists(sections_data_dir):
            shutil.rmtree(sections_data_dir)
            print(f'Delete Old Directory {sections_data_dir}')

        ### Write sections/ again
        os.makedirs(sections_data_dir)
        start_process_raw(args.data_dir, sections_data_dir, args.select_relations)

    # Create Splits
    if args.split == 'L2_xval':
        pdtb3_make_splits_xval(sections_data_dir, args.output_dir, level='L2')
    elif args.split == 'L1_ji':
        pdtb3_make_splits_l1(sections_data_dir, args.output_dir)
    elif args.split == 'L2_community':
        pdtb3_make_community(sections_data_dir, args.output_dir, level='L2')
    else:
        raise ValueError('--split must be one of "L2_xval", "L3_xval", "L1_ji".')





if __name__ == '__main__':
    main()