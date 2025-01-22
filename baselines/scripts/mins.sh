export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

# num_epochs=200
# n_init=200
# batch_size=100
# max_evals=10000

# for task in Ackley Rastrigin Levy Rosenbrock; do
#     for dim in 200 400; do
#         for seed in 0 1 2 3; do
#             # seed를 바탕으로 device를 순환 (0 -> 1, 1 -> 2, 2 -> 3, 3 -> 1, ...)
#             device=$(( (seed % 3) + 1 ))
#             CUDA_VISIBLE_DEVICES=${device} \
#             python baselines/algorithms/mins.py --task $task --dim $dim \
#                    --n_init $n_init --batch_size $batch_size --max_evals $max_evals \
#                    --seed $seed --num_epochs $num_epochs &
#         done
#         wait
#     done
#     wait
# done
# wait


num_epochs=200
n_init=100
batch_size=50
max_evals=2000

for seed in 0 1 2 3; do
    device=$(( (seed % 2) + 2 ))
    CUDA_VISIBLE_DEVICES=${device} \
    python baselines/algorithms/mins.py --task RoverPlanning --dim 100 \
            --n_init $n_init --batch_size $batch_size --max_evals $max_evals \
            --seed $seed --num_epochs $num_epochs &
done
wait

num_epochs=200
n_init=100
batch_size=50
max_evals=2000

for seed in 0 1 2 3; do
    device=$(( (seed % 2) + 2 ))
    CUDA_VISIBLE_DEVICES=${device} \
    python baselines/algorithms/mins.py --task DNA --dim 180 \
            --n_init $n_init --batch_size $batch_size --max_evals $max_evals \
            --seed $seed --num_epochs $num_epochs &
done
wait

num_epochs=200
n_init=100
batch_size=50
max_evals=2000

for seed in 0 1 2 3; do
    device=$(( (seed % 2) + 2 ))
    CUDA_VISIBLE_DEVICES=${device} \
    python baselines/algorithms/mins.py --task HalfCheetah --dim 102 \
            --n_init $n_init --batch_size $batch_size --max_evals $max_evals \
            --seed $seed --num_epochs $num_epochs &
done
wait