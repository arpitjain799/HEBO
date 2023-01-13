#!/bin/bash

TASKS=( ackley-53 )

# ------------------------------------------------------------
# BEGIN .34
task_name="ackley-53"
task_name="xgboost_opt"

PYTHON_SCRIPT=./experiments/mix_and_match/mix_and_match_mixed_any.py

models=(gp_to gp_o gp_diffusion)
acq_opts=(ga is sa)

TASKSET_START=30
for use_tr in 0 1; do
  if (( use_tr == 0 )); then use_tr=''; else use_tr='--use_tr'; fi
  taskset_end=$(( TASKSET_START - 1 ))
  for i in $(seq 0 $((${#models[@]} - 1))); do
    model=${models[i]}
    device_id=$(( i % 4))
    device_id=-1

    taskset_start=$(( taskset_end + 1 ))
    taskset_end=$(( taskset_start + 5 ))

    optimizers_ids=""
    for j in $(seq 0 $((${#acq_opts[@]} - 1))); do
      acq_opt=${acq_opts[j]}
      optimizers_ids="${optimizers_ids} ${model}_${acq_opt}"
    done

    cmd="taskset -c ${taskset_start}-${taskset_end} python ${PYTHON_SCRIPT} --task_name $task_name --optimizers_ids ${optimizers_ids} $use_tr --device_id ${device_id}"
    echo $cmd
    $cmd &
  done
  wait
done

taskset -c 25-29 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name aig_optimization_hyp --optimizers_ids gp_o_ga gp_o_sa --use_tr --device_id 0 --seeds 42 43 44 45 46
taskset -c 30-34 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name aig_optimization_hyp --optimizers_ids gp_o_is gp_o_mab --use_tr --device_id 2 --seeds 42 43 44 45 46


# END .34
# ------------------------------------------------------------


# ------------------------------------------------------------
# BEGIN .35
task_name="aig_optimization_hyp"

PYTHON_SCRIPT=./experiments/mix_and_match/mix_and_match_mixed_any.py

models=(gp_to gp_o gp_diffusion)
acq_opts=(ga is sa)

TASKSET_START=30
for use_tr in 0 1; do
  if (( use_tr == 0 )); then use_tr=''; else use_tr='--use_tr'; fi
  taskset_end=$(( TASKSET_START - 1 ))
  for i in $(seq 0 $((${#models[@]} - 1))); do
    model=${models[i]}
    device_id=$(( i % 4))
    device_id=-1

    taskset_start=$(( taskset_end + 1 ))
    taskset_end=$(( taskset_start + 5 ))

    optimizers_ids=""
    for j in $(seq 0 $((${#acq_opts[@]} - 1))); do
      acq_opt=${acq_opts[j]}
      optimizers_ids="${optimizers_ids} ${model}_${acq_opt}"
    done

    cmd="taskset -c ${taskset_start}-${taskset_end} python ${PYTHON_SCRIPT} --task_name $task_name --optimizers_ids ${optimizers_ids} $use_tr --device_id ${device_id}"
    echo $cmd
    $cmd &
  done
  wait
done

taskset -c 0-4   python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name aig_optimization_hyp --optimizers_ids gp_to_is --use_tr --device_id 0 --seeds 46
taskset -c 5-9   python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name aig_optimization_hyp --optimizers_ids gp_o_ga gp_to_mab --device_id 3 --seeds 42 43 44 45 46
taskset -c 10-14 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name aig_optimization_hyp --optimizers_ids gp_to_sa gp_to_mab gp_o_sa --use_tr --device_id 2 --seeds 42 43 44 45 46
taskset -c 15-19 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name aig_optimization_hyp --optimizers_ids gp_to_is --device_id 2 --seeds 44 45 46

scp -r /home/antoineg/Projects/combopt_release/results/EDA\ Sequence\ Optimization\ -\ Design\ sin\ -\ Ops\ basic\ -\ Pattern\ basic_w_post_map\ -\ Hyps\ boils_hyp_op_space\ -\ Obj\ both antoine@10.227.91.34:/home/antoine/Projects/combopt_lib_release/results/


# END .35
# ------------------------------------------------------------

# ------------------------------------------------------------
# BEGIN .23
task_name="ackley-53"

PYTHON_SCRIPT=./experiments/mix_and_match/mix_and_match_mixed_any.py

models=(gp_to gp_o gp_diffusion)
acq_opts=(ga is sa)

TASKSET_START=30
for use_tr in 0 1; do
  if (( use_tr == 0 )); then use_tr=''; else use_tr='--use_tr'; fi
  taskset_end=$(( TASKSET_START - 1 ))
  for i in $(seq 0 $((${#models[@]} - 1))); do
    model=${models[i]}
    device_id=$(( i % 4))
    device_id=-1

    taskset_start=$(( taskset_end + 1 ))
    taskset_end=$(( taskset_start + 5 ))

    optimizers_ids=""
    for j in $(seq 0 $((${#acq_opts[@]} - 1))); do
      acq_opt=${acq_opts[j]}
      optimizers_ids="${optimizers_ids} ${model}_${acq_opt}"
    done

    cmd="taskset -c ${taskset_start}-${taskset_end} python ${PYTHON_SCRIPT} --task_name $task_name --optimizers_ids ${optimizers_ids} $use_tr --device_id ${device_id}"
    echo $cmd
    $cmd &
  done
  wait
done

taskset -c 0-4   python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_ga --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 5-9   python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_ga --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 10-14 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_sa --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 15-19 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_sa --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 20-24 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_is --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 25-29 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_is --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 30-34 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_mab --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 35-39 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_to_mab --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 40-44 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_ga --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 45-49 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_ga --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 50-54 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_sa --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 55-59 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_sa --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 60-64 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_is --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 65-69 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_is --use_tr --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 70-74 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_mab --device_id -1 --seeds 42 43 44 45 46 &
taskset -c 75-79 python ./experiments/mix_and_match/mix_and_match_mixed_any.py --task_name ackley-53 --optimizers_ids gp_o_mab --use_tr --device_id -1 --seeds 42 43 44 45 46 &

scp -r /home/rladmin/antoineg/Projects/combopt_release/results/Ackley\ Function\ 50-nom-2\ 3-num antoine@10.227.91.34:/home/antoine/Projects/combopt_lib_release/results/
# END .23
# ------------------------------------------------------------
