#!/bin/bash

TASKS=( ackley aig_optim_basic antibody_design mig_optim pest_control rna_inverse_fold )
TASKS=( ackley aig_optim_basic pest_control rna_inverse_fold )

HEAD=./experiments/benchmark_baselines/benchmark_baselines_

for task_id in $(seq 0 $((${#TASKS[@]} - 1))); do
  task=${TASKS[task_id]}
  cmd="taskset -c 25-30 python $HEAD${task}.py"
  echo $cmd
  $cmd
done