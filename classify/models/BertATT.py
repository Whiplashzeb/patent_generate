from transformers import BertPreTrainedModel, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


class BertATT(BertPreTrainedModel):
    def __init__(self, config):
        super(BertATT, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        last_hidden_states = outputs[0]
        last_hidden_states = self.dropout(last_hidden_states)

        score = torch.tanh(torch.matmul(last_hidden_states, self.W_w))
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        scored_x = last_hidden_states * attention_weights
        feat = torch.sum(scored_x, dim=1)

        logits = self.classifier(feat)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
