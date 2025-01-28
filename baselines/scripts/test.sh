#Synthetic

for seed in 1; do
   CUDA_VISIBLE_DEVICES=4 python baselines/algorithms/dibo.py --task Ackley --dim 200 --batch_size 100\
       --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
       --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30 --buffer_size 2000 &
done

wait
