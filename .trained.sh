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

# DNC v0.0.4 512-2-20.73-19.17
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 10 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-bptt-500.pt --log-interval 20 --bptt 500

# DNC v0.0.5 512-2x2-38.05-34.99
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-dncv0.0.5-cuda1.pt --log-interval 20

# num hidden layers = 1 512-2x1-45.44-42.28
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 2 --nr_cells 8 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-dncv0.0.5-cuda1.pt --log-interval 20

python main.py --nhid 512 --nlayers 4 --emsize 512 --cell_size 512 --read_heads 2 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-4-dncv0.0.5-cuda0.pt --log-interval 20

python main.py --nhid 512 --nlayers 4 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --log-interval 20


# visdom discussion - 52.29-48.72
python main.py --nhid 512 --debug --nlayers 2 --nhlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --log-interval 20

python ./generate.py --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --words 20 --cuda 1 --outf generated.txt

# layers=4 not done
# controller layers
for (( i = 2; i < 10; i++ )); do
  python main.py --nhid 512 --nlayers ${i} --nhlayers 2 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-${i}-dncv0.0.5-cuda1.pt --log-interval 20 > PTB-512-layers-${i}-dncv0.0.5-cuda1.log
done

# 1,2,3 done
# controller layer hidden layers
for (( i = 4; i < 5; i++ )); do
  python main.py --nhid 512 --nlayers 2 --nhlayers ${i} --emsize 512 --cell_size 256 --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x${i}-dncv0.0.5-cuda0.pt --log-interval 20 > PTB-512-layers-2x${i}-dncv0.0.5-cuda0.log
done

# stopped at 14
# memory cells
for (( i = 2; i < 20; i++ )); do
  python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells $((i*2)) --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x2-cells-$((i*2))-cuda0.pt --log-interval 20 > PTB-512-layers-2x2-cells-$((i*2))-cuda0.log
done

# memory cell size
for (( i = 1; i < 7; i++ )); do
  python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size $(( 512 / 2**$i )) --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-cuda0.pt --log-interval 20 > PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-debug-cuda0.log
done

# memory cell size vs nr cells
for (( i = 1; i < 7; i++ )); do
  python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size $(( 512 / 2**$i )) --read_heads 4 --nr_cells $((2**$i*2)) --cell_size $(( 512 / 2**$i )) --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-nr_cells-$((2**$i*2))-cuda0.pt --log-interval 20 > PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-nr_cells-$((2**$i*2))-cuda0.log
done
