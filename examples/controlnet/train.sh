MODEL_NAME="/graphics/scratch2/students/grosskop/diffusers/examples/instruct_pix2pix/output_08_13_addNoise"
PROJECT_NAME="Thesis ControlNet + InstructPix2Pix"
RUN_TITLE="25_01_04 Canny Edges Loss"
RUN_DESCRIPTION="ControlNet of finetuned InstructPix2Pix using H&E Image as input to frozen network and Canny Edges of H&E and H&E Lantes to ControlNet and Canny Edges loss between predicted image and H&E Input Image"
OUTPUT_DIR="${RUN_TITLE// /_}"

export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

nohup accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES train_controlnet.py \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name="$PROJECT_NAME" \
    --name="$RUN_TITLE" \
    --description="$RUN_DESCRIPTION" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_batch_size=4 \
    --learning_rate=5e-6 \
    --num_train_epochs=100 \
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
    