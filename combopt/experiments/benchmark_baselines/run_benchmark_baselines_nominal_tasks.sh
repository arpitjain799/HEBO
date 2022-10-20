#!/bin/bash

TASKS=( ackley aig_optim_basic antibody_design mig_optim pest_control rna_inverse_fold )

HEAD=./experiments/benchmark_baselines/benchmark_baselines_

for task_id in $(seq 0 $((${#TASKS[@]} - 1))); do
  task=${TASKS[task_id]}
  cmd="taskset -c 25-30 python $HEAD${task}.py"
  echo $cmd
  $cmd
done

# taskset -c 5-9 python experiments/benchmark_baselines/benchmark_baselines_antibody_design.py --device_id 1 --absolut_dir /home/kamild/projects/combopt/libs/Absolut2/src/AbsolutNoLib
# taskset -c 10-14 python experiments/mix_and_match/mix_and_match_antibody_design.py --device_id 0 --absolut_dir /home/kamild/projects/combopt/libs/Absolut3/src/AbsolutNoLib
# taskset -c 15-19 python experiments/mix_and_match/mix_and_match_antibody_design.py --use_tr --device_id 1 --absolut_dir /home/kamild/projects/combopt/libs/Absolut4/src/AbsolutNoLib
