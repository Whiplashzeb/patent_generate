from transformers import BertPreTrainedModel, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

class BertLSTM(BertPreTrainedModel):
    def __init__(self, config, rnn_hidden_size):
        super(BertLSTM, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        self.classifier = nn.Linear(rnn_hidden_size * 2, self.num_labels)

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
        _, (hidden, cell) = self.rnn(last_hidden_states)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        logits = self.classifier(hidden)

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
