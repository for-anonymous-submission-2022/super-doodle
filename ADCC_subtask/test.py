import torch
import time

class TCTest:
    def test(model, dataloader, device, label_reversed_dict):
        r"""Validation function to evaluate model performance on a 
        separate set of data.

        This function will return the true and predicted labels so we can use later
        to evaluate the model's performance.

        This function is built with reusability in mind: it can be used as is as long
            as the `dataloader` outputs a batch in dictionary format that can be passed 
            straight into the model - `model(**batch)`.

        Arguments:

            dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
                Parsed data into batches of tensors.

            device_ (:obj:`torch.device`):
                Device used to load tensors before feeding to model.

        Returns:
            
            :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
                Labels, Train Average Loss]
        """
        
        predictions_labels = []
        true_labels = []

        reversed_true_labels = []
        reversed_pred_labels = []

        total_loss = 0

        model.eval()
        
        start = time.time()

        for i, batch in enumerate(dataloader):
            true_labels = batch['labels'].numpy().flatten().tolist()

            batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

            with torch.no_grad():        
                outputs = model(**batch)

                loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                total_loss += loss.item()

                predict_content = logits.argmax(axis=-1).flatten().tolist()

                predictions_labels = predict_content

                if i % 50 == 0:
                    end_batch = time.time()
                    print(f"---------- Time taken until batch {i}/{len(dataloader)}: {end_batch - start:.5f} sec ----------")

            reversed_true_label = []
            reversed_pred_label = []
            for idx, label in enumerate(true_labels):
                if label != -100:
                    reversed_true_label.append(label_reversed_dict[label])
                    reversed_pred_label.append(label_reversed_dict[predictions_labels[idx]])
            if len(reversed_true_label) > 1:
                reversed_true_labels.append(reversed_true_label)
                reversed_pred_labels.append(reversed_pred_label)

        end = time.time()
        print(f"---------- Time taken for test: {end - start:.5f} sec ----------")

        return reversed_true_labels, reversed_pred_labels