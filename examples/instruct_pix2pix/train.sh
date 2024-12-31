MODEL_NAME="botp/stable-diffusion-v1-5"
PROJECT_NAME="New Instruct Pix2Pix"
RUN_TITLE="24_12_31 Vanilla MSE + Offset Noise"
RUN_DESCRIPTION="This is the vanilla Instruct Pix2Pix config which should server as a benchmark"
OUTPUT_DIR="${RUN_TITLE// /_}"

export CUDA_VISIBLE_DEVICES=3
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

nohup accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES train_instruct_pix2pix.py \
    --output_dir=$OUTPUT_DIR \
    --project="$PROJECT_NAME" \
    --name="$RUN_TITLE" \
    --description="$RUN_DESCRIPTION" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_batch_size=16 \
    --learning_rate=1e-4\
    --num_train_epochs=500 \
    --mixed_precision="bf16" \
    --resolution=512 \
    --translation_prompt="Transform H&E-stained tissue, featuring pink cytoplasm and blue nuclei, into ER (IHC) stained tissue with brown ER-positive nuclei and light pink counterstained background." \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --seed=0 \
    --report_to="wandb" \
    --checkpointing_steps=2000 \
    --checkpoints_total_limit=4 \
    > $OUTPUT_DIR.log 2>&1 &