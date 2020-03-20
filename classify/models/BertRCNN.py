from transformers import BertPreTrainedModel, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class BertRCNN(BertPreTrainedModel):
    def __init__(self, config, rnn_hidden_size, layers, dropout):
        super(BertRCNN, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.W = Linear(config.hidden_size + 2 * rnn_hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

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

        pooled_output = outputs[1]

        rnn_output, _ = self.rnn(last_hidden_states)

        x = torch.cat((rnn_output, last_hidden_states), 2)
        y = torch.tanh(self.W(x)).permute(0, 2, 1)
        y = F.max_pool1d(y, y.size()[2]).squeeze(2)

        feature = torch.cat([pooled_output, y], dim=-1)

        logits = self.classifier(feature)
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
