#!/usr/bin/env bash
NUM_LAYERS='2 3 4 5'
NUM_UNITS='16 12 10'
BOTTLENECK='10 8 6 4 3'
LEARNING_RATE='0.1 0.01 0.001 0.0001'

for num_layers in $NUM_LAYERS
do
for num_hidden_units in $NUM_UNITS
do
for num_bottleneck_units in $BOTTLENECK
do
for learning_rate in $LEARNING_RATE
do
python autoencoder.py --num_layers $num_layers --num_hidden_units $num_hidden_units --num_bottleneck_units $num_bottleneck_units --learning_rate $learning_rate --train_dir 'numLayers'$num_layers'_numHiddenUnits'$num_hidden_units'_bottleneckUnits'$num_bottleneck_units'_learningRate'$learning_rate
done
done
done
done
