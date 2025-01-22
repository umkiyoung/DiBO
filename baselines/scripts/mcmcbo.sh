export PYTHONPATH=/home/uky/repos_python/Research/DIBO:$PYTHONPATH

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/mcmcbo.py --task Ackley --dim 200 --tr_num 1 --batch_size 100\
#        --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/mcmcbo.py --task Ackley --dim 400 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

# RASTRIGIN

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/mcmcbo.py --task Rastrigin --dim 200 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/mcmcbo.py --task Rastrigin --dim 400 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# # wait

# # Levy

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/mcmcbo.py --task Levy --dim 200 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# # wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/mcmcbo.py --task Levy --dim 400 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

# Rosenbrock

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=2 python baselines/algorithms/mcmcbo.py --task Rosenbrock --dim 200 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/mcmcbo.py --task Rosenbrock --dim 400 --tr_num 1 --batch_size 100\
#         --noise_var 0 --n_init 200 --max_evals 6000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

# NOTE: MUJOCO

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/mcmcbo.py --task HalfCheetah --dim 102 --tr_num 1 --batch_size 50\
#        --noise_var 0 --n_init 100 --max_evals 2000 --seed $seed --use_mcmc MH --repeat_num 1
# done

# wait

for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/mcmcbo.py --task RoverPlanning --dim 100 --tr_num 1 --batch_size 50\
         --noise_var 0 --n_init 100 --max_evals 2000 --seed $seed --use_mcmc MH --repeat_num 1
    done

#DNA
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/mcmcbo.py --task DNA --dim 180 --tr_num 1 --batch_size 50\
         --noise_var 0 --n_init 100 --max_evals 2000 --seed $seed --use_mcmc MH --repeat_num 1
    done