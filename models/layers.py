import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM", bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bias = bias

        if rnn_type == "LSTM":
            self.rnn_cell = nn.LSTMCell(input_size, hidden_size, bias)
        else:
            raise NotImplementedError

    def forward(self, input, state=None):

        batch_size = input.shape[0]
        sequence_length = input.shape[1]
        if state is not None:
            h, c = state
        else:
            h = torch.zeros(batch_size, self.hidden_size).to(input.device)
            c = torch.zeros(batch_size, self.hidden_size).to(input.device)

        h_list = []
        for i in range(sequence_length):
            h, c = self.rnn_cell(input[:,i,:], (h, c))
            h_list.append(h)
        h = torch.stack(h_list, 1)
        return h, (h_list[-1], c)

    def __repr__(self):
        return "{} (input_size={}, hidden_size={}, bias={})".format(
            self.rnn_type,
            self.input_size,
            self.hidden_size,
            self.bias
        )

class Sequence2SequenceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, decoder_type="LSTM", bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder_type = decoder_type
        self.bias = bias

        self.hidden_to_input = nn.Linear(hidden_size, input_size)

        if decoder_type == "LSTM":
            self.decoder = nn.LSTMCell(input_size, hidden_size, bias=bias)
        else:
            raise NotImplementedError

    def forward(self, x, encoder_output_h):
        # input:
        #       x: batch_size, sequence_length, hidden_size
        #       encoder_output_h: batch_size, input_size
        # output:
        #       decoder_output: batch_size, sequence_length, hidden_size

        batch_size, sequence_length, hidden_size = x.shape

        hx = encoder_output_h
        cx = torch.zeros_like(hx).to(encoder_output_h.device)
        input = torch.zeros(batch_size, self.input_size).to(encoder_output_h.device)

        decoder_output = []
        for i in range(sequence_length):
            try:
                hx, cx = self.decoder(input, (hx, cx))
            except:
                import pdb; pdb.set_trace()
            input = self.hidden_to_input(hx.detach())
            decoder_output.append(hx)
        # 列表反转
        # 解码的第一个是原始序列的最后一个 反向解码
        decoder_output = decoder_output[::-1]
        decoder_output = torch.stack(decoder_output, 1)

        return decoder_output

    def __repr__(self):
        return "{} (input_size={}, hidden_size={}, bias={})".format(
            self.decoder_type,
            self.input_size,
            self.hidden_size,
            self.bias
        )



