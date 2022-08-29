import re

#import spacy
#nlp = spacy.load("en_core_web_sm")

class Stage1_Processor:
    def __init__(self):
        pass

    def run(self, text):
        tokenized = self.tokenize(text)

        return{
            'tokenized' : tokenized
        }

    def tokenize(self, text):
        tokenized = text.strip().split()
        #doc = nlp(text)
        #token_list = [token.text for token in doc]
        #tokenized = [re.sub(' ', '', token) for token in token_list]
        #tokenized = list(filter(lambda x: x != '', tokenized))

        return tokenized

class Stage2_Processor:
    def __init__(self):
        pass
    
    def run(self, input_ADCC, output_ADCC):
        if "B-Conn" in output_ADCC:
            self.next_stage = "EDSC"
        else:
            self.next_stage = "NDSC"
        
        remain = self.remove_O(input_ADCC, output_ADCC)
        paired = self.pair_tokens_and_tags(input_ADCC, output_ADCC)

        return {
            "remain"      : remain,
            "next_stage"  : self.next_stage,
            "paired"      : paired,
            "tokens_only" : input_ADCC,
            "tags_only"   : output_ADCC
        }

    def remove_O(self, input_ADCC, output_ADCC):
        remain = []
        for idx, tag in enumerate(output_ADCC):
            if tag != "O":
                remain.append(input_ADCC[idx])
        
        return remain
    
    def pair_tokens_and_tags(self, input_ADCC, output_ADCC):
        paired = []
        for idx, token in enumerate(input_ADCC):
            paired.append((token, output_ADCC[idx]))
        
        return paired