import json

import ADCC_subtask.inference as ADCC
import EDRC_subtask.inference as EDSC
import NDRC_subtask.inference as NDSC
from processor import Stage1_Processor
from processor import Stage2_Processor

from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer

from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score       as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

def model_name_processor(model_name):
    if '/' in model_name:
        return model_name
    if model_name.lower() ==  "edits-bert-base":
        return 'ADCC_Community_BERT_base_v_1_0_0'

class parser:
    def __init__(self, model_nameA ,model_nameB ,model_nameC):
        self.prepare_stage1(model_nameA)
        self.prepare_stage2(model_nameB, model_nameC)
    
    def prepare_stage1(self, model_nameA):
        model_name = model_name_processor(model_nameA)
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_nameA.split('_')[-3], 
            num_labels=7
            )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_nameA.split('_')[-3]
            )
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_nameA, config=model_config
            )
        self.Stage1_Processor        = Stage1_Processor() 
        self.Stage1_Classifier_ADCC  = ADCC.Prediction(model, tokenizer)
    
    def prepare_stage2(self, model_nameB, model_nameC):
        model_nameB = model_name_processor(model_nameB)
        model_configB = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_nameB.split('_')[-3], 
            num_labels=26
            )
        tokenizerB = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_nameB.split('_')[-3]
            )
        modelB = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_nameB, config=model_configB
            )
        model_configC = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_nameC.split('_')[-3], 
            num_labels=26
            )
        tokenizerC = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_nameC.split('_')[-3]
            )
        modelC = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_nameC, config=model_configC
            )
        self.Stage2_Processor   = Stage2_Processor() 
        self.Stage2_Classifier_EDSC  = EDSC.Prediction(modelB, tokenizerB)
        self.Stage2_Classifier_NDSC  = NDSC.Prediction(modelC, tokenizerC)

    def run(self, raw):
        stage2_processed, input_stage2, output_stage2 = self.pipeline(raw)
        
        return {
            **stage2_processed,
            'sense': output_stage2
        }

    def pipeline(self, raw):
        stage1_processed = self.Stage1_Processor.run(raw)
        input_ADCC, output_ADCC = \
            self.Stage1_Classifier_ADCC.run(stage1_processed)

        stage2_processed = self.Stage2_Processor.run(input_ADCC, output_ADCC)
        if stage2_processed["next_stage"] == "EDSC":
            input_EDSC, output_EDSC = \
                self.Stage2_Classifier_EDSC.run(stage2_processed)
            return stage2_processed, input_EDSC, output_EDSC
        elif stage2_processed["next_stage"] == "NDSC":
            input_NDSC, output_NDSC = \
                self.Stage2_Classifier_NDSC.run(stage2_processed)
            return stage2_processed, input_NDSC, output_NDSC

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
        accuracy_scores.append(seq_accuracy_score([label], [prediction]))
        f1_scores.append(seq_f1_score([label], [prediction]))
        recall_scores.append(seq_recall_score([label], [prediction]))
        precision_scores.append(seq_precision_score([label], [prediction]))
    return sum(accuracy_scores)/len(accuracy_scores), sum(f1_scores)/len(f1_scores), sum(recall_scores)/len(recall_scores), sum(precision_scores)/len(precision_scores)

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

if __name__ == "__main__":
    import tqdm
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

    to_run = [
        ("bert-base-uncased", "exp-L2-14way"),
        ("bert-base-uncased", "nexp-L2-14way+EntRel"),
        ("roberta-base", "exp-L2-14way"),
        ("roberta-base", "nexp-L2-14way+EntRel")
        ]
    for tuple_item in to_run:
        total_stage1_accuracy = []
        total_stage1_f1       = []
        total_stage1_p        = []
        total_stage1_r        = []
        total_stage2_accuracy = []
        total_stage2_f1       = []
        total_stage2_p        = []
        total_stage2_r        = []
        total_end_accuracy    = []
        for fold in range(1,13):
            model_used = tuple_item[0]
            data_used = tuple_item[1]
            EDiTS = parser(
                model_nameA = f'ADCC_subtask/checkpoint/ADCCall+EntRel_2e-06_bert-base-uncased_fold_{fold}.pt',
                model_nameB = f'EDRC_subtask/checkpoint/EDRCexp-L2-14way_2e-06_{model_used}_fold_{fold}.pt',
                model_nameC = f'NDRC_subtask/checkpoint/NDRCnexp-L2-14way+EntRel_2e-06_{model_used}_fold_{fold}.pt'
            )
            data = pd.read_json(f'PDTB_dataset/EDiTS_datasets/{data_used}/fold_{fold}/test.json', orient='records')

            print(model_used)
            print(data_used)

            data_dict = data.to_dict('records')

            skips = 1
            total_end = 1
            correct_end = 1
            sense_correct = 1
            adcc_correct = 1
            pred_tags_list = []
            true_tags_list = []
            pred_sense_list = []
            true_sense_list = []
            pair_sense_list = []
            for row_idx,row in enumerate(data_dict):
                try:
                    
                    stage1_correct = False
                    stage2_correct = False

                    sent_tokens = row['sent_tokens']
                    sent_tags = row['sent_tags']
                    discourse_relations = EDiTS.run(raw = sent_tokens)
                    sent_tokens = sent_tokens.strip().split()
                    sent_tags = sent_tags.strip().split(',')
                    pred_tags = discourse_relations['tags_only']

                    assert len(sent_tokens) == len(pred_tags), f"{sent_tokens}, {pred_tags} sth wrong with lengths"

                    pred_tags_list.append(pred_tags)
                    true_tags_list.append(sent_tags)

                    match = 0
                    total = 0
                    for idx,tag in enumerate(sent_tags):
                        total += 1
                        if tag == pred_tags[idx]:
                            match += 1
                    if match/total > 0.7:
                        stage1_correct = True
                        adcc_correct += 1

                    true_sense = row['sense1']
                    pair_sense = row['sense2']
                    pred_sense = discourse_relations['sense']
                    pred_sense_list.append(pred_sense)
                    true_sense_list.append(true_sense)
                    pair_sense_list.append(pair_sense)

                    if pred_sense == true_sense or pred_sense == pair_sense:
                        stage2_correct = True
                        sense_correct += 1

                    total_end += 1
                    if stage1_correct == True and stage2_correct == True:
                        correct_end += 1
                    
                except:
                    skips += 1
                #    print('_'*10)
                #    print(discourse_relations['tokens_only'])
                #    print(len(discourse_relations['tokens_only']))
                #    print(sent_tokens)
                #    print(len(sent_tokens))
                if row_idx % 50 == 0:
                    print(f"steps: {row_idx}/{len(data_dict)}")
                    print(f"mid-skip: {skips}")
                    print(f"mid-sense:{sense_correct/total_end}")
                    print(f"mid-adcc: {adcc_correct/total_end}")
                    print(f"mid-end: {correct_end/total_end}")
            stage1_accuracy, stage1_F1, stage1_precision, stage1_recall = partial_scores(true_tags_list, pred_tags_list)
            stage2_accuracy, stage2_F1, stage2_precision, stage2_recall = partial_scoring(true_sense_list, pred_sense_list, pair_sense_list)
            print(model_used)
            print(data_used)
            print("-"*10+f"fold:{fold}")
            print(f"skip: {skips}")
            print(f"{fold}'s stage1_ACC: {stage1_accuracy}")
            print(f"{fold}'s stage1_F1 : {stage1_F1}")
            print(f"{fold}'s stage1_F1 : {stage1_precision}")
            print(f"{fold}'s stage1_F1 : {stage1_recall}")
            print(f"{fold}'s stage2_ACC: {stage2_accuracy}")
            print(f"{fold}'s stage2_F1 : {stage2_F1}")
            print(f"{fold}'s stage2_F1 : {stage2_precision}")
            print(f"{fold}'s stage2_F1 : {stage2_recall}")
            print(f"{fold}'s end-to-end: {correct_end/total_end}")
            total_stage1_accuracy.append(stage1_accuracy)
            total_stage1_f1      .append(stage1_F1)
            total_stage1_p       .append(stage1_precision)
            total_stage1_r       .append(stage1_recall)
            total_stage2_accuracy.append(stage2_accuracy)
            total_stage2_f1      .append(stage2_F1)
            total_stage2_p       .append(stage2_precision)
            total_stage2_r       .append(stage2_recall)
            total_end_accuracy   .append(correct_end/total_end)
        print(model_used)
        print(data_used)
        print(f"{fold}'s stage1_ACC: {sum(total_stage1_accuracy)/len(total_stage1_accuracy)}")
        print(f"{fold}'s stage1_F1 : {sum(total_stage1_f1)/len(total_stage1_f1)}")
        print(f"{fold}'s stage1_p : {sum(total_stage1_p)/len(total_stage1_p)}")
        print(f"{fold}'s stage1_r : {sum(total_stage1_r)/len(total_stage1_r)}")
        print(f"{fold}'s stage2_ACC: {sum(total_stage2_accuracy)/len(total_stage2_accuracy)}")
        print(f"{fold}'s stage2_F1 : {sum(total_stage2_f1)/len(total_stage2_f1)}")
        print(f"{fold}'s stage2_p : {sum(total_stage2_p)/len(total_stage2_p)}")
        print(f"{fold}'s stage2_r : {sum(total_stage2_r)/len(total_stage2_r)}")
        print(f"{fold}'s end_to_end: {sum(total_end_accuracy)/len(total_end_accuracy)}")
