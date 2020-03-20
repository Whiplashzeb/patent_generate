from transformers import BertPreTrainedModel, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BertBiDAF(BertPreTrainedModel):
    def __init__(self, config, rnn_hidden_size, dropout):
        super(BertBiDAF, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # contextual embedding layer
        self.question_LSTM = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=dropout)
        self.context_LSTM = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=dropout)

        # attention flow layer
        self.att_weight_c = nn.Linear(rnn_hidden_size * 2, 1)
        self.att_weight_q = nn.Linear(rnn_hidden_size * 2, 1)
        self.att_weight_cq = nn.Linear(rnn_hidden_size * 2, 1)

        # modeling layer
        self.model_LSTM1 = nn.LSTM(rnn_hidden_size * 8, rnn_hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=dropout)
        self.model_LSTM2 = nn.LSTM(rnn_hidden_size * 2, rnn_hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=dropout)

        # output layer
        self.start_weight_flow = nn.Linear(rnn_hidden_size * 8, 1)
        self.start_weight_model = nn.Linear(rnn_hidden_size * 2, 1)
        self.end_weight_flow = nn.Linear(rnn_hidden_size * 8, 1)
        self.end_weight_model = nn.Linear(rnn_hidden_size * 2, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]  # [batch, seq_len, bert_dim=768]

        question, (hidden, cell) = self.question_LSTM(sequence_output)  # [batch, seq_len, hidden_size * 2]
        context, (hidden, cell) = self.context_LSTM(sequence_output)  # [batch, seq_len, hidden_size * 2]

        attn_flow = self.att_flow_layer(question, context)

        context_model, (hidden, cell) = self.model_LSTM1(attn_flow)
        context_model, (hidden, cell) = self.model_LSTM2(context_model)  # [batch, seq_len, hidden_size * 2]

        start_logits, end_logits = self.output_layer(attn_flow, context_model)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            startloss = loss_fct(start_logits, start_positions)
            endloss = loss_fct(end_logits, end_positions)
            total_loss = (startloss + endloss) / 2
            outputs = (total_loss,) + outputs

        return outputs

    def att_flow_layer(self, question, context):
        """
        question : (batch, seq_len, hidden_size * 2)
        context : (batch, seq_len, hidden_size * 2)
        return : (batch, seq_len, seq_len)
        """

        seq_len = question.size(1)

        cq = list()
        for i in range(seq_len):
            qi = question.select(1, i).unsqueeze(1)  # [batch, 1, hidden_size * 2]
            ci = self.att_weight_cq(context * qi).squeeze()  # [batch, seq_len, 1]
            cq.append(ci)
        cq = torch.stack(cq, dim=-1)  # [batch, seq_len, seq_len]

        # [batch, seq_len, seq_len]
        s = self.att_weight_c(context).expand(-1, -1, seq_len) + self.att_weight_q(question).permute(0, 2, 1).expand(-1, seq_len, -1) + cq

        a = F.softmax(s, dim=2)  # [batch, seq_len, seq_len]
        c2q_att = torch.bmm(a, question)  # [batch, seq_len, seq_len] * [batch, seq_len, hidden_size * 2] -> [batch, c_len, hidden_size * 2]

        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)  # [batch, 1, seq_len]
        q2c_att = torch.bmm(b, context).squeeze()  # [batch, 1, seq_len] * [batch, seq_len, hidden_size * 2] -> [batch, 1, hidden_size * 2]
        q2c_att = q2c_att.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_size * 2]

        x = torch.cat([context, c2q_att, context * c2q_att, context * q2c_att], dim=-1)  # [batch, seq_len, hidden_size * 8]

        return x

    def output_layer(self, attn_flow, context_model):
        """
        attn_flow : [batch, seq_len, hidden_size * 8]
        context_model : [batch, seq_len, hidden_size * 2]
        return start_logits : [batch, seq_len]; end_logits : [batch, seq_len]
        """
        start_logits = (self.start_weight_flow(attn_flow) + self.start_weight_model(context_model)).squeeze()  # [batch, seq_len]
        end_logits = (self.end_weight_flow(attn_flow) + self.end_weight_model(context_model)).squeeze()  # [batch, seq_len]

        return start_logits, end_logits
