MODEL_NAME="botp/stable-diffusion-v1-5"
PROJECT_NAME="Thesis InstructPix2Pix"
RUN_TITLE="25_02_14 DEBUGGING"
RUN_DESCRIPTION="baseline"
OUTPUT_DIR="${RUN_TITLE// /_}"

export CUDA_VISIBLE_DEVICES=3
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

nohup accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES train_instruct_pix2pix_jointly.py \
    --num_train_epochs=300 \
    --validation_epochs=1 \
    --prediction_type="epsilon" \
    --input_perturbation=0.1 \
    --snr_gamma=5 \
    --bias_ihc_he=0.5 \
    --conditioning="input" \
    --output_dir=$OUTPUT_DIR \
    --project="$PROJECT_NAME" \
    --name="$RUN_TITLE" \
    --description="$RUN_DESCRIPTION" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_batch_size=16 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=1000 \
    --mixed_precision="bf16" \
    --resolution=512 \
    --translation_prompt="IHC" \
    --he_generation_prompt="H&E" \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --seed=0 \
    --checkpointing_steps=10000 \
    --checkpoints_total_limit=4 \
    > $OUTPUT_DIR.log 2>&1 &

    # --report_to="wandb" \
    # --conditioning    choices=["input", "xattention", "combined"],