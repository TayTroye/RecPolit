# 固定参数
MODEL_NAME=LEDRec
DATASET=TMALL

LR_LIST=(0.001)
LOG_PREFIX="tmall_debug_sft"


for lr in "${LR_LIST[@]}"; do
    LOG_FILE="${LOG_PREFIX}_lr${lr}.csv"

    echo "Running lr=${lr}, log_file=${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
      --main_process_port 40000 \
      --num_processes 4 start_sft.py \
      --dataset=${DATASET} \
      --config_file=models/${MODEL_NAME}/tm2w_sft.yaml \
      --search_k=40 \
      --lr=${lr} \
      --epochs=100 \
      --patience=10 \
      --purchase_loss_weight=5 \
      --temperature_sample=1 \
      --top_p_sample=1 \
      --num_heads=2 \
      --log_file=${LOG_FILE} \
      --squeeze_data=True
done



