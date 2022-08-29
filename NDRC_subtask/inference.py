import torch

labels_to_ids =\
    {
        'Expansion.Conjunction'       :0   ,
        'Comparison.Concession'       :1   ,
        'Contingency.Cause'           :2   ,
        'Temporal.Asynchronous'       :3   ,
        'Temporal.Synchronous'        :4   ,
        'Contingency.Condition'       :5   ,
        'Comparison.Contrast'         :6   ,
        'Expansion.Manner'            :7   ,  
        'Contingency.Purpose'         :8   ,
        'Expansion.Instantiation'     :9   ,
        'Expansion.Level-of-detail'   :10  ,
        'Expansion.Substitution'      :11  ,
        'Expansion.Disjunction'       :12  ,
        'Contingency.Neg-condition'   :13  ,
        'Comparison.Similarity'       :14  ,
        'Contingency.Cond+SpeechAct'  :15  ,
        'Contingency.Cause+Belief'    :16  ,
        'Expansion.Exception'         :17  ,
        'Expansion.Equivalence'       :18  ,
        'Comparison.Conc+SpeechAct'   :19  ,
        'Contingency.Cause+SpeechAct' :20  ,
        'Contingency.Negative-cause'  :21  , 
        'NotCon'                      :22  ,
        'NotMat'                      :23  , 
        'EntRel'                      :24  ,
        'None'                        :25
    }

# prediction to text label
ids_to_labels = {v:k for k, v in labels_to_ids.items()}

class Prediction:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def run(self, stage2_processed):
        remain = stage2_processed['remain']
        self.input_NDSC  = []
        self.output_NDSC = []

        inputs = self.process_inputs(remain)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            prediction = logits.argmax(axis=-1).flatten().tolist()
            self.output_NDSC = ids_to_labels[prediction[0]]
        
        return self.input_NDSC, self.output_NDSC
    
    def process_inputs(self, remain):
        self.input_NDSC = ' '.join(remain)
        inputs = self.tokenizer(
          self.input_NDSC, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length=256
            )
        
        return inputs