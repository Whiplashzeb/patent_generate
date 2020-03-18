from transformers import BertPreTrainedModel, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


class BertDPCNN(BertPreTrainedModel):
    def __init__(self, config, filter_num):
        super(BertDPCNN, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conv_region = nn.Conv2d(1, filter_num, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(filter_num, filter_num, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fn = nn.ReLU()
        self.classifier = nn.Linear(filter_num, config.num_labels)

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
        last_hidden_states = last_hidden_states.unsqueeze(1)

        x = self.conv_region(last_hidden_states)
        x = self.padding_conv(x)
        x = self.act_fn(x)

        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.act_fn(x)

        x = self.conv(x)
        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.squeeze()

        logits = self.classifier(x)
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

    def _block(self, x):
        x = self.padding_pool(x)
        px = self.pooling(x)
        x = self.padding_conv(px)
        x = F.relu(x)

        x = self.conv(x)
        x = self.padding_conv(x)
        x = F.relu(x)

        x = self.conv(x)
        x = x + px
        return x