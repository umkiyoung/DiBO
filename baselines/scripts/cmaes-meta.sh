export PYTHONPATH=/home/uky/repos_python/Research/DIBO:$PYTHONPATH
#Ackley Rastrigin Levy Rosenbrock
for task in Ackley; do
    for dim in 200; do
        for seed in 0 1 2 3; do
            device=$((1))
            CUDA_VISIBLE_DEVICES=$device python baselines/algorithms/cma-meta-algorithm/test-main.py --task $task --dim $dim --batch_size 1\
                 --n_init 200 --max_evals 10000 --seed $seed --solver turbo
        done
    done
done

wait
