
#! /bin/bash


for mbs in 1 10 50; do
  for gs in 10 15 20; do
    sbatch --export=MBS=$mbs,GS=$gs,MT=0,SG=0 train_cluster.sbatch
    sleep 1
  done
done
