export PYTHONPATH=/home/uky/repos_python/Research/DIBO:$PYTHONPATH

# for task in Ackley; do
#     for buffer_size in 500 2000 3000 4000; do
#         if [ $buffer_size -eq 500 ]; then
#             device=0
#         elif [ $buffer_size -eq 2000 ]; then
#             device=1
#         elif [ $buffer_size -eq 3000 ]; then
#             device=2
#         elif [ $buffer_size -eq 4000 ]; then
#             device=3
#         fi
#         CUDA_VISIBLE_DEVICES=$device python baselines/algorithms/dibo.py --task $task --dim 200 --batch_size 100\
#             --n_init 200 --max_evals 10000 --seed 0 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size $buffer_size --abalation TEST &
        
#     done
#     wait
# done

# wait

# for task in Rastrigin Levy Rosenbrock; do
#     for buffer_size in 500 1000 2000 3000; do
#         if [ $buffer_size -eq 500 ]; then
#             device=0
#         elif [ $buffer_size -eq 1000 ]; then
#             device=1
#         elif [ $buffer_size -eq 2000 ]; then
#             device=2
#         elif [ $buffer_size -eq 3000 ]; then
#             device=3
#         fi
#         CUDA_VISIBLE_DEVICES=$device python baselines/algorithms/dibo.py --task $task --dim 200 --batch_size 100\
#             --n_init 200 --max_evals 10000 --seed 0 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --local_search True --alpha 1e-5 --local_search_epochs 30 --diffusion_steps 30 --buffer_size $buffer_size --abalation TEST &
        
#     done
#     wait
# done


# for dim in 200 400; do
#     if [ $dim -eq 200 ]; then
#         device=1
#     elif [ $dim -eq 400 ]; then
#         device=2
#     fi
#     CUDA_VISIBLE_DEVICES=$device python baselines/algorithms/dibo.py --task Rastrigin --dim $dim --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed 0 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30 --buffer_size 2000 --abalation TEST &

# done
# # wait

# for dim in 200 400; do
#     if [ $dim -eq 200 ]; then
#         device=0
#     elif [ $dim -eq 400 ]; then
#         device=3
#     fi
#     CUDA_VISIBLE_DEVICES=$device python baselines/algorithms/dibo.py --task Levy --dim $dim --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed 0 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-5 --local_search_epochs 15 --diffusion_steps 30 --buffer_size 500 --abalation TEST &

# done
# wait





#-----------------------------

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task Ackley --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30 --buffer_size 2000
# done

# wait
# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/dibo.py --task Ackley --dim 400 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
#        --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30 --buffer_size 2000 --proxy_hidden_dim 512
# done


# wait
# # RASTRIGINs

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-4 --local_search_epochs 50 --diffusion_steps 30
# done

# wait

for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/dibo.py --task Rastrigin --dim 400 --batch_size 100\
        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
        --local_search True --alpha 1e-5 --local_search_epochs 15 --diffusion_steps 50 --buffer_size 2000 --proxy_hidden_dim 512
done

# wait

# # Levy

# for seed in 0; do
#     for ls in 10; do
#         CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/dibo.py --task Levy --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-5 --local_search_epochs $ls --diffusion_steps 30 --buffer_size 1000
#     done
# done

# # wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=2 python baselines/algorithms/dibo.py --task Levy --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
#         --local_search True --alpha 1e-5 --local_search_epochs 5 --diffusion_steps 30 --buffer_size 2000
# done

# wait

# # Rosenbrock

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task Rosenbrock --dim 200 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-4 --local_search_epochs 50 --diffusion_steps 30
# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task Rosenbrock --dim 400 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 200 --num_prior_epochs 250 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30

# done

# wait

# NOTE: MUJOCO

# for seed in 2; do
#     CUDA_VISIBLE_DEVICES=2 python baselines/algorithms/dibo.py --task Walker2d --dim 102 --batch_size 100\
#         --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 200 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 5e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 500 --gamma 1.0 --proxy_hidden_dim 512

# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/dibo.py --task HalfCheetah --dim 102 --batch_size 100\
#         --n_init 1000 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-4 --local_search_epochs 30 --diffusion_steps 30 --buffer_size 500

# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/dibo.py --task Ant --dim 102 --batch_size 100\
#         --n_init 1000 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-4 --local_search_epochs 30 --diffusion_steps 30 --buffer_size 500

# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0 python baselines/algorithms/dibo.py --task Humanoid --dim 102 --batch_size 100\
#         --n_init 1000 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-4 --local_search_epochs 30 --diffusion_steps 30 --buffer_size 500

# done

# wait
# NOTE: LunarLanding

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task LunarLanding --dim 12 --batch_size 100\
#         --n_init 200 --max_evals 1000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30

# done

# for seed in {0..4}; do
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task HalfCheetah --dim 102 \
#                                                                --batch_size 50 --n_init 100 --max_evals 2000 --seed $seed \
#                                                                --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
#                                                                --local_search True --alpha 1e-4 --local_search_epochs 10 --gamma 1.0 --buffer_size 300 &
# done
# wait

# NOTE: HalfCheetah
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/dibo.py --task HalfCheetah --dim 102 \
#                                                                --batch_size 50 --n_init 100 --max_evals 2000 --seed $seed \
#                                                                --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
#                                                                --local_search True --alpha 1e-4 --local_search_epochs 15 --gamma 1.0 --buffer_size 300 
# done
# NOTE: RoverPlanning

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task RoverPlanning --dim 100 \
#                                                                --batch_size 50 --n_init 100 --max_evals 2000 --seed $seed \
#                                                                --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
#                                                                --local_search True --alpha 1e-5 --local_search_epochs 30 --gamma 1.0 --buffer_size 300 

# done

# wait

# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/algorithms/dibo.py --task RoverPlanning --dim 100 \
#                                                                --batch_size 50 --n_init 100 --max_evals 2000 --seed $seed \
#                                                                --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
#                                                                --local_search True --alpha 1e-5 --local_search_epochs 10 --gamma 1.0 --buffer_size 300 

# done


# wait
# # NOTE: DNA

# for seed in 2 3; do
#     for epoch in  30 ; do
#         CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/dibo.py --task DNA --dim 180 \
#                                                                 --batch_size 50 --n_init 100 --max_evals 2000 --seed $seed \
#                                                                 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
#                                                                 --local_search True --alpha 1e-5 --local_search_epochs $epoch --gamma 1.0 --buffer_size 300
#     done
# done