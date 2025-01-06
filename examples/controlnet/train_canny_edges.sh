MODEL_NAME="/graphics/scratch2/students/grosskop/diffusers/examples/instruct_pix2pix/output_08_13_addNoise"
PROJECT_NAME="Thesis ControlNet + InstructPix2Pix"
RUN_TITLE="25_01_05 MSE and Canny Edges MSE on Prediction and HE"
RUN_DESCRIPTION="MSE is computed between added noise to latents and predicted noise. Canny edges MSE is computed between the predicted IHC Pixel Image and ground truth H&E pixel image."
# Alternative non-linear weight scheduling: w = (1 - t/T)^2, Linear weight scheduling: w = - t/T + 1

OUTPUT_DIR="${RUN_TITLE// /_}"

export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

nohup accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES train_controlnet_canny_edges.py \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name="$PROJECT_NAME" \
    --name="$RUN_TITLE" \
    --description="$RUN_DESCRIPTION" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_batch_size=1 \
    --learning_rate=1e-6 \
    --num_train_epochs=50 \
    --mixed_precision="bf16" \
    --resolution=512 \
    --translation_prompt="Transform H&E-stained tissue, featuring pink cytoplasm and blue nuclei, into ER (IHC) stained tissue with brown ER-positive nuclei and light pink counterstained background." \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --seed=0 \
    --report_to="wandb" \
    --validation_steps=2048 \
    --checkpointing_steps=8192 \
    --checkpoints_total_limit=4 \
    > $OUTPUT_DIR.log 2>&1 &
    