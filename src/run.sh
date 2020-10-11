#!bin/sh

export PYTHONPATH=$PYTHONPATH:/Users/deepudilip/ML/ml_framework/

python train.py --fold 0 --model extratrees
python train.py --fold 1 --model extratrees
python train.py --fold 2 --model extratrees
python train.py --fold 3 --model extratrees
python train.py --fold 4 --model extratrees

