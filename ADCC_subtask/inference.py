import torch

labels_to_ids = {
    'O': 0, 'B-Arg1': 1, 'I-Arg1': 2, 'B-Arg2': 3, 'I-Arg2': 4, 'B-Conn': 5, 'I-Conn': 6
    }
ids_to_labels = {k: v for v, k in labels_to_ids.items()}

class Prediction:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()


    def run(self, stage1_processed):
        tokenized = stage1_processed['tokenized']
        self.input_ADCC = []
        self.len_input_ADCC = 0
        self.output_ADCC = []
        
        inputs = self.process_inputs(tokenized)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            prediction = logits.argmax(axis=-1).flatten().tolist()
            for tag in prediction[1:self.len_input_ADCC+1]:
                self.output_ADCC.append(ids_to_labels[tag])

        self.post_process(tokenized)

        assert len(self.input_ADCC) == len(self.output_ADCC), 'something wrong with lengths'
        #assert len(self.input_ADCC) == len(tokenized), 'something wrong with lengths'

        return self.input_ADCC, self.output_ADCC


    def process_inputs(self, tokenized):
        inputs = self.tokenizer(
            [tokenized], is_split_into_words = True, add_special_tokens = True, truncation = True, padding = 'max_length', return_tensors = 'pt', max_length = 256
            )
        for token in self.tokenizer.convert_ids_to_tokens(inputs.input_ids.tolist()[0]):
            if token not in ['[CLS]','[SEP]','[PAD]']:
                self.input_ADCC.append(token)
                self.len_input_ADCC += 1

        return inputs


    def post_process(self, tokenized):
        processed_input_ADCC  = []
        processed_output_ADCC = []

        for token, tag in zip(self.input_ADCC, self.output_ADCC):
            if "##" in token:
                processed_input_ADCC[-1] += token[2:]
            else:
                processed_input_ADCC.append(token)
                processed_output_ADCC.append(tag)
        
        idx1 = 0
        idx2 = 0
        new_processed_input_ADCC = []
        new_processed_output_ADCC = []
        while len(new_processed_input_ADCC) != len(tokenized):
            if processed_input_ADCC[idx1] == tokenized[idx2].lower():
                new_processed_input_ADCC.append(processed_input_ADCC[idx1])
                new_processed_output_ADCC.append(processed_output_ADCC[idx1])
                idx1 += 1
                idx2 += 1
            else:
                new_processed_input_ADCC.append(processed_input_ADCC[idx1])
                new_processed_output_ADCC.append(processed_output_ADCC[idx1])
                while processed_input_ADCC[idx1] in tokenized[idx2].lower():
                    new_processed_input_ADCC[-1]+=processed_input_ADCC[idx1]
                    idx1 += 1
                idx2 += 1


        self.input_ADCC  = new_processed_input_ADCC 
        self.output_ADCC = new_processed_output_ADCC
                    

    #def post_process(self, tokenized):
    #    processed_input_ADCC = self.input_ADCC
    #    processed_output_ADCC = self.output_ADCC
    #    for idx, token in enumerate(self.input_ADCC):
    #        print(token)
    #        if idx < len(tokenized):
    #            if token != tokenized[idx]:
    #                if '##' in token:
    #                    print(token)
    #                    processed_input_ADCC[idx-1] = \
    #                        processed_input_ADCC[idx-1] + processed_input_ADCC[idx][2:]
    #                   del processed_input_ADCC[idx]
    #                   del processed_output_ADCC[idx]
    #               if token == tokenized[idx].lower():
    #                   processed_input_ADCC[idx] = tokenized[idx]
    #               if token + processed_input_ADCC[idx+1] == tokenized[idx].lower():
    #                    processed_input_ADCC[idx] = tokenized[idx]
    #                    del processed_input_ADCC[idx+1]
    #                    del processed_output_ADCC[idx+1]
    #    self.input_ADCC = processed_input_ADCC
    #    self.output_ADCC = processed_output_ADCC

if __name__ == "__main__":
    model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased', 
            num_labels=7
            )
    model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path='checkpoint/ADCCall+EntRel_2e-06_bert-base-uncased_fold_1.pt', config=model_config
            )