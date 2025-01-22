export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Ackley --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Ackley --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# # RASTRIGIN

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Rastrigin --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# # Levy

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Levy --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Levy --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# # Rosenbrock

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Rosenbrock --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task Rosenbrock --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# NOTE: MUJOCO

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/cmaes.py --task Walker2d --dim 102 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/cmaes.py --task HalfCheetah --dim 102 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/cmaes.py --task Ant --dim 888 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/cmaes.py --task Humanoid --dim 6392 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# NOTE: LunarLanding

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/cmaes.py --task LunarLanding --dim 12 --batch_size 100\
#         --n_init 200 --max_evals 1000 --seed $seed
# done

# wait

NOTE: RoverPlanning
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task RoverPlanning --dim 100 --batch_size 50\
        --n_init 100 --max_evals 2000 --seed $seed
done

NOTE: DNA

for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/cmaes.py --task DNA --dim 180 --batch_size 50\
        --n_init 100 --max_evals 2000 --seed $seed
done