k=10
export LD_LIBRARY_PATH=/miniconda3/envs/hiob/lib/

conda run -n hiob /rust-search/target/release/rust-search \
            -i data -o result \
            --its 300000 \
            --k 10 \
            --size 10M \
            --bits 1024 \
            --noise 0.09 \
            --probe-min 50 \
            --probe-max 60 \
            --probe-steps 2

conda activate hiob
          rust-search/target/release/rust-search \
            -i data -o result \
            --its 300000 \
            --k $k \
            --size 30M \
            --bits 1024 \
            --noise 0.07 \
            --probe-min 60 \
            --probe-max 70 \
            --probe-steps 2

conda run -n hiob rust-search/target/release/rust-search \
            -i data -o result \
            --its 300000 \
            --k $k \
            --size 100M \
            --bits 1024 \
            --noise 0.05 \
            --probe-min 70 \
            --probe-max 80 \
            --probe-steps 2
