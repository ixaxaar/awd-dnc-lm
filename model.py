import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from dnc import SDNC

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        nhlayers,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        wdrop=0,
        tie_weights=False,
        nr_cells=5,
        read_heads=2,
        sparse_reads=10,
        cell_size=10,
        gpu_id=-1,
        independent_linears=False,
        debug=True
    ):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.debug = debug
        assert rnn_type in ['LSTM', 'QRNN', 'DNC', 'SDNC'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else ninp, save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True if l != nlayers - 1 else True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        elif rnn_type.lower() == 'sdnc':
            self.rnns = []
            self.rnns.append(
                SDNC(
                    input_size=ninp,
                    hidden_size=nhid,
                    num_layers=nlayers,
                    num_hidden_layers=nhlayers,
                    rnn_type='lstm',
                    nr_cells=nr_cells,
                    read_heads=read_heads,
                    sparse_reads=sparse_reads,
                    cell_size=cell_size,
                    gpu_id=gpu_id,
                    independent_linears=independent_linears,
                    debug=debug,
                    dropout=0
                )
            )
        elif rnn_type.lower() == 'dnc':
            self.rnns = []
            self.rnns.append(
                DNC(
                    input_size=ninp,
                    hidden_size=nhid,
                    num_layers=nlayers,
                    num_hidden_layers=nhlayers,
                    rnn_type='lstm',
                    nr_cells=nr_cells,
                    read_heads=read_heads,
                    cell_size=cell_size,
                    gpu_id=gpu_id,
                    independent_linears=independent_linears,
                    debug=debug,
                    dropout=wdrop
                )
            )
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, ntoken)

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

    def forward(self, input, hidden, return_h=False, reset_experience=True):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        if self.debug:
            debug_mems = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            if 'dnc' in self.rnn_type.lower():
                raw_output = raw_output.transpose(0, 1)
                if self.debug:
                    raw_output, new_h, debug = rnn(raw_output, hidden[l], reset_experience=reset_experience, pass_through_memory=True)
                    debug_mems.append(debug)
                else:
                    raw_output, new_h = rnn(raw_output, hidden[l], reset_experience=reset_experience)
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
            if self.debug:
                return result, hidden, raw_outputs, outputs, debug_mems
            return result, hidden, raw_outputs, outputs
        if self.debug:
            return result, hidden, debug_mems
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
        elif 'dnc' in self.rnn_type.lower():
            return [None]
