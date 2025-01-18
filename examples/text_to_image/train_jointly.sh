#"botp/stable-diffusion-v1-5"
MODEL_NAME="/graphics/scratch2/students/grosskop/diffusers_thesis/examples/instruct_pix2pix/25_01_02_MSE_+_Offset_Noise"
PROJECT_NAME="Thesis TextToImage"
RUN_TITLE="25_01_18 MSE Joint"
RUN_DESCRIPTION="Vanilla MSE + Offset Noise + Cosine LR 1e-4"
OUTPUT_DIR="${RUN_TITLE// /_}"

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

nohup accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES train_text_to_image_jointly.py \
    --output_dir=$OUTPUT_DIR \
    --project="$PROJECT_NAME" \
    --name="$RUN_TITLE" \
    --description="$RUN_DESCRIPTION" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_batch_size=16 \
    --learning_rate=1e-4 \
    --translation_prompt="IHC" \
    --he_generation_prompt="H&E" \
    --lr_warmup_steps=100 \
    --lr_scheduler="cosine" \
    --num_train_epochs=10 \
    --noise_offset=0.1 \
    --validation_epochs=1 \
    --mixed_precision="bf16" \
    --resolution=512 \
    --translation_prompt="Transform H&E-stained tissue, featuring pink cytoplasm and blue nuclei, into ER (IHC) stained tissue with brown ER-positive nuclei and light pink counterstained background." \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --seed=0 \
    --checkpointing_steps=5000 \
    --checkpoints_total_limit=4 \
    > $OUTPUT_DIR.log 2>&1 &

#   --report_to="wandb" \    
#   --use_ema \
#   --max_grad_norm=1 \
