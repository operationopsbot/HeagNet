#!/bin/bash
for i in {0..9}
do
   # python isoforest_experiment.py --name telecom-large --id $i 
   python train_telecom_large.py --id $i 
done