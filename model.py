import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from dnc import DNC

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        wdrop=0,
        tie_weights=False,
        nr_cells=5,
        read_heads=2,
        cell_size=10,
        gpu_id=-1,
        independent_linears=True
    ):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'DNC'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else ninp, save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True if l != nlayers - 1 else True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        elif rnn_type.lower() == 'dnc':
            if nhid != ninp:
                raise ValueError('When using DNC units, nhid must be equal to emsize')
            self.rnns = []
            self.rnns.append(
                DNC(
                    'lstm',
                    hidden_size=nhid,
                    num_layers=nlayers,
                    nr_cells=nr_cells,
                    read_heads=read_heads,
                    cell_size=cell_size,
                    gpu_id=gpu_id,
                    independent_linears=independent_linears
                )
            )
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            if self.rnn_type.lower() == 'dnc':
                raw_output = raw_output.transpose(0, 1)
                raw_output, new_h = rnn(raw_output, hidden[l], reset_experience=True)
                raw_output = raw_output.transpose(0, 1)
            else:
                raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout).contiguous()
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type.lower() == 'dnc':
            return [None]
