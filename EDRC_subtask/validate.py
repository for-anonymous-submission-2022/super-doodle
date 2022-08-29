import torch
import time

class GeneralValidation:
    def validation(model, dataloader, device, pair: bool = False):
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

        # Tracking variables
        predictions_labels = []
        true_labels = []
        true_labels_pair = []
        #total loss for this epoch.
        total_loss = 0

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        
        start = time.time()
        # Evaluate data for one epoch
        for i, batch in enumerate(dataloader):
            # add original labels
            true_labels += batch['labels'].numpy().flatten().tolist()
            if pair:
                true_labels_pair += batch['labels_pair'].numpy().flatten().tolist()
                del batch['labels_pair']

            # move batch to device
            batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(**batch)

                # The call to `model` always returns a tuple, so we need to pull the 
                # loss value out of the tuple along with the logits. We will use logits
                # later to to calculate training accuracy.
                loss, logits = outputs[:2]
                
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_loss += loss.item()
                
                # get predicitons to list
                predict_content = logits.argmax(axis=-1).flatten().tolist()

                # update list
                predictions_labels += predict_content

                if i % 50 == 0:
                    end_batch = time.time()
                    print(f"---------- Time taken until batch {i}/{len(dataloader)}: {end_batch - start:.5f} sec ----------")

        end = time.time()
        print(f"---------- Time taken for validation: {end - start:.5f} sec ----------")

        # Calculate the average loss over the training data.
        avg_epoch_loss = total_loss / len(dataloader)

        # Return all true labels and prediciton for future evaluations.
        if pair:
            return true_labels, true_labels_pair, predictions_labels, avg_epoch_loss

        return true_labels, predictions_labels, avg_epoch_loss