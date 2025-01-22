
export PYTHONPATH=/home/uky/repos_python/Research/DIBO:$PYTHONPATH


for seed in {0..4}; do
    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/dibo.py --task HalfCheetah --dim 102 \
                                                               --batch_size 50 --n_init 100 --max_evals 2000 --seed $seed \
                                                               --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
                                                               --local_search True --alpha 1e-4 --local_search_epochs 10 --gamma 1.0 --buffer_size 300 &
done
wait