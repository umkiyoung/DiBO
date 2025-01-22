export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Ackley --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Ackley --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_epochs 400
# done

# wait

# RASTRIGIN

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Rastrigin --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_epochs 400
# done

# wait

# # Levy

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Levy --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed 
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Levy --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_epochs 400
# done

# wait

# # Rosenbrock

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Rosenbrock --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Rosenbrock --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_epochs 400
# done

# wait

# # NOTE: MUJOCO

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Walker2d --dim 102 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task HalfCheetah --dim 102 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Ant --dim 888 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task Humanoid --dim 6392 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed
# done

# wait

# #NOTE: Luna Landing

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/diffbbo.py --task LunarLanding --dim 12 --batch_size 100\       
#    --n_init 200 --max_evals 1500 --seed $seed
# done

#NOTE: RoverPlanning

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/diffbbo.py --task RoverPlanning --dim 100 --batch_size 50\
#          --n_init 100 --max_evals 2000 --seed $seed
#     done

# wait

#NOTE: DNA

for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/diffbbo.py --task DNA --dim 180 --batch_size 50\
         --n_init 100 --max_evals 2000 --seed $seed
    done