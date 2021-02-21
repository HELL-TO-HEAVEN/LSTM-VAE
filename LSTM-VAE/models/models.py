import torch
import torch.nn as nn

from models.layers import Sequence2SequenceDecoder, RNNModel

class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.encoder = RNNModel(input_size=input_size, hidden_size=hidden_size, rnn_type="LSTM")
        self.mean_layer = nn.Linear(hidden_size, hidden_size)
        self.log_var_layer = nn.Linear(hidden_size, hidden_size)
        self.encoder_to_decoder = nn.Linear(hidden_size, input_size)
        self.decoder = Sequence2SequenceDecoder(input_size=hidden_size, hidden_size=input_size, bias=True, decoder_type="LSTM")

    def forward(self, x):
        # input:
        #       x: batch_size, sequence_length, input_size
        # output:
        #       reconstruct_x: batch_size, sequence_length, input_size

        _, (hx, _) = self.encoder(x)
        mean = self.mean_layer(hx)
        log_var = self.log_var_layer(hx)
        # reparameter
        z = self.reparameter(mean, log_var)
        z = self.encoder_to_decoder(z.squeeze(0))
        reconstruct_x = self.decoder(x, z)

        return reconstruct_x

    def reparameter(self, mean, log_var):
        eps = torch.randn(mean.shape).to(mean.device)
        z = mean + torch.sqrt(torch.exp(log_var)) * eps
        return z

    def encode(self, x):
        # hx: batch_size, hidden_size
        _,(hx, _) = self.encoder(x)
        return hx

    def encode_and_variational(self, x):
        hx = self.encode(x)
        mean = self.mean_layer(hx)
        log_var = self.log_var_layer(hx)
        return mean, log_var

    def decode(self, x, z):
        return self.decoder(x, z)
