export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1


DATA_LIST=(Musical_Instruments Industrial_and_Scientific Video_Games)

DATA_LIST=(Industrial_and_Scientific Video_Games)


for DATA in ${DATA_LIST[@]}; do

    python main.py \
        --dataset=$DATA \
        --config_file=config.yaml \
        --ckpt_name=rqvae


    python generate_tokens.py \
        --dataset=$DATA \
        --config_file=config.yaml \
        --ckpt_name=rqvae

done

