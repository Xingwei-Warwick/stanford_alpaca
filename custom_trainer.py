from transformers import Trainer
import torch
from torch import nn
import torch.nn.functional as F


IGNORE_INDEX = [-100, 1, 2]


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits_ids = logits.argmax(dim=-1)

        splitted_labels = []
        for i in range(labels.size(0)):
            this_labels_list = []
            start = 0
            for j in range(labels.size(1)):
                if labels[i][j] == 29871:
                    this_labels_list.append(labels[i][start:j])
                    start = j + 1
            this_labels_list.append(labels[i][start:])
            splitted_labels.append(this_labels_list)

        loss = 0.
        for i in range(logits_ids.size(0)):
            start = 0
            count = 0
            all_output = []
            output_ind_list = []
            for j in range(logits_ids.size(1)):
                if logits_ids[i][j] not in IGNORE_INDEX:
                    output_ind_list.append(str(logits_ids[i][j]))

                if logits_ids[i][j] == 29871:
                    all_output.append(' '.join(output_ind_list))
                    output_ind_list = []

                    min_loss = 1000.
                    for k in range(len(splitted_labels[i])):
                        this_loss = self.loss_fct(logits[i][start:j], F.pad(splitted_labels[i][k],(0,logits[i][start:j].size(0)-splitted_labels[i][k].size(0)), mode='constant', value=0))
                        if this_loss<min_loss:
                            min_loss = this_loss
                    loss += min_loss
                    start = j + 1
                    count += 1

                min_loss = 1000.
                for k in range(len(splitted_labels[i])):
                    #print(logits[i][start:].size(), splitted_labels[i][k])
                    this_loss = self.loss_fct(logits[i][start:], F.pad(splitted_labels[i][k],(0,logits[i][start:].size(0)-splitted_labels[i][k].size(0)), mode='constant', value=0))
                    if this_loss<min_loss:
                        min_loss = this_loss
                loss += min_loss
                loss /= count + 1
            
            # compute repeatative penalty
            repeat_entries = len(all_output) - len(set(all_output))
            loss += repeat_entries * 10 # repeatative penalty coefficient
            loss *= 0.01

        # \n is 29871

        # compute custom loss (suppose one has 3 labels with different weights)
        #loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    


