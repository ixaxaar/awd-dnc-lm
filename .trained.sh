#!/usr/bin/env bash


# fate - 400-37.19-34.70
python main.py --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB.pt

# 512x2-30.89-28.29
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

python main.py --nhid 512 --nlayers 10 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 10 --batch_size 30 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512x10.pt --log-interval 20

# no reset experience 512x2-49.92-
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

# LSTM-512x2-85.31-79.95
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model LSTM --epoch 50 --cuda 0 --save PTB-LSTM-512x2.pt --log-interval 20


# with lr annealing - does not work
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

# 512-2-25.69-24.45
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 50 --data data/wikitext-2 --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --log-interval 20 --save WT2-512.pt


python ./generate.py --data data/penn --checkpoint PTB-512x2-30.89-28.29.pt --words 1000 --cuda --outf generated.txt

python ./generate.py --data data/wikitext-2 --checkpoint WT2-512-2-25.69-24.45.pt --words 1000 --cuda --outf generated-wikitext-2.txt


