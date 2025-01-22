export PYTHONPATH=/home/uky/repos_python/Research/DIBO:$PYTHONPATH

# # # TODO: Add the abalation for the Main Components A.2.1 Priority 1
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation Default&
# done
# wait
# for seed in 0 1 2 3; do # No Filtering
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering False --abalation MainComponents&
# done
# wait
# for seed in 0 1 2 3; do # No Filtering, Local Search
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --buffer_size 1000 --alpha 1e-5 --local_search False --local_search_epochs 0 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering False --abalation MainComponents&
# done
# wait
# for seed in 0 1 2 3; do #No Filtering, Local Search, Posterior training
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#         --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 0\
#         --buffer_size 1000 --alpha 1e-5 --local_search False --local_search_epochs 0 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering False --abalation MainComponents&
# done
# wait

# # #TODO: Add the abalation for the Reweighting Scheme A.1.1 Priority 1
# for reweighting in uniform; do #value rank; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting $reweighting --filtering True --abalation Reweighting&
#     done
#     wait
# done
# wait

# #TODO: Add the abalation for the alpha A.2.2 Priority 1
# for alpha in 1e-4 1e-3 1e-2 1e-1 1e0; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 1000 --alpha $alpha --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation Alpha&
#     done
#     wait
# done
# wait

# #TODO: Add the abalation for the buffer size A.3.1 Priority 1
# for buffer_size in 200 500 2000 5000 10000; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size $buffer_size --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation Buffer&
#     done
#     wait
# done
# wait

# #TODO: Add the abalation for the uncertainty estimation A.1.3 Priority 2
for uncertainty_estimation in dropout None; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
            --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
            --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
            --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --uncertainty_estimation $uncertainty_estimation\
            --abalation UncertaintyEstimation&
    done
    wait
done
wait

# #TODO: Add the ablation for the gamma A.1.4 Priority 2
for gamma in 0 0.1 0.5 3.0 5.0 10.0; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
            --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
            --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
            --gamma $gamma --num_ensembles 5 --reweighting exp --filtering True --abalation Gamma&
    done
    wait
done
wait

# #TODO: Add the abalation for the local search epochs A.2.3 Priority 2
# for local_search_epochs in 0 30 50 100; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs $local_search_epochs --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation LocalSearchEpochs&
#     done
#     wait
# done
# wait

#TODO: Add the abalation for the Amortized Sampler A.2.5 Priority 2
for training_posterior in on off; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
            --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
            --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
            --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --training_posterior $training_posterior\
            --abalation AmortizedSampler&
    done
    wait
done
wait

# #TODO: Add the abalation for the Experiment Setting A.3.2 Priority 2
# for n_init in 100 500 1000 2000; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init $n_init --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation NInit&
#     done
#     wait
# done
# wait

# for batch_size in 10 50 200 400 800; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size $batch_size\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation BatchSize&
#     done
#     wait
# done
# wait

# # #TODO: Add the abalation for the training steps A.1.2 Priority 3
# for numepoch in 100 150; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs $numepoch --num_prior_epochs $numepoch\
#             --num_posterior_epochs $numepoch --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10\
#             --diffusion_steps 30 --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation TrainingSteps&
#     done
#     wait
# done
# wait

# #TODO: Add the abalation for the diffusion steps A.3.3 Priority 3
# for diffusion_steps in 10 50 100; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task Rastrigin --dim 200 --batch_size 100\
#             --n_init 200 --seed $seed --max_evals 10000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 1000 --alpha 1e-5 --local_search True --local_search_epochs 10 --diffusion_steps $diffusion_steps\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --abalation DiffusionSteps&
#     done
#     wait
# done
# wait
