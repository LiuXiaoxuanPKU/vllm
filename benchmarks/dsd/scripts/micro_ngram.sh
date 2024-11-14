TARGET="lmsys/vicuna-7b-v1.5"
input_len=256
WARMUP=5
REPEAT=10

for batch_size in 1 2 4 8 16 32 64 128
do
    python benchmarks/benchmark_latency.py \
        --model  $TARGET \
        --disable-async-output-proc \
        --input-len $input_len \
        --output-len 256 \
        --batch-size $batch_size \
        --num-iters-warmup $WARMUP \
        --num-iters $REPEAT \
        --output-json benchmarks/dsd/results/llama2-7b-ngram/org_bz=${batch_size}_input-len=${input_len}.json
done


acc=0.7
for match in 0.9
do
    for batch_size in 1 2 4 8 16 32 64 128
    do
        python benchmarks/benchmark_latency.py \
            --model  $TARGET \
            --input-len $input_len \
            --output-len 256 \
            --batch-size $batch_size \
            --num-iters-warmup $WARMUP \
            --num-iters $REPEAT \
            --speculative-model "[ngram]" \
            --ngram-prompt-lookup-min 2 \
            --ngram-prompt-lookup-max 8 \
            --num-speculative-tokens 7 \
            --dsd \
            --acceptance-rate $acc \
            --dummy-match $match \
            --output-json benchmarks/dsd/results/llama2-7b-ngram/dsd_bz=${batch_size}_input-len=${input_len}_acc=${acc}_match=${match}.json
    done


    for batch_size in 1 2 4 8 16 32 64
    do
        for num_speculative_tokens in 1 3 5 7
        do
            python benchmarks/benchmark_latency.py \
                --model  $TARGET \
                --input-len $input_len \
                --output-len 256 \
                --batch-size $batch_size \
                --num-iters-warmup $WARMUP \
                --num-iters $REPEAT \
                --speculative-model "[ngram]" \
                --ngram-prompt-lookup-min 2 \
                --ngram-prompt-lookup-max 8 \
                --num-speculative-tokens $num_speculative_tokens \
                --acceptance-rate $acc \
                --dummy-match $match \
                --output-json benchmarks/dsd/results/llama2-7b-ngram/vsd=${num_speculative_tokens}_bz=${batch_size}_input-len=${input_len}_acc=${acc}_match=${match}.json
        done
    done
done