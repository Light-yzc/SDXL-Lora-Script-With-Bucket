支持XL的lora训练和dream booth lora训练的分桶数据集
---
目前只测试了 `train_text_to_image_lora_sdxl.py`, dream booth等以后再测试修改
---
分桶使用方法：
加入参数
```bash
--enable_bucketing --buckets="768x1280,896x1120,1024x1024,1280x768" 
```
例如：
```bash
!accelerate launch /content/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="YOUR PATH" \
  --train_data_dir="DATASET_DIR" \
  --output_dir="OUTPUT_DIR" \
  --resume_from_checkpoint="RESUME_DIR" \
  --caption_column="text" \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=3 \
  --num_train_epochs=15 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=7e-5 \
  --max_train_steps=4000 \
  --lr_scheduler="cosine" \
  --snr_gamma=5.0 \
  --lr_warmup_steps=100 \
  --use_8bit_adam \
  --mixed_precision="bf16" \
  --rank=128 \
  --enable_bucketing --buckets="768x1280,896x1120,1024x1024, 1280x768" \ #add bucket
  --validation_prompt="kotalu_v1, 1girl, solo, large black rose headwear, feather headdress, floral crown, black beaded chains, dangling star charms, gothic aesthetic, portrait shot" \
  --validation_epochs=3 \
  --checkpointing_steps=200 \
  --checkpoints_total_limit=5 \
  --report_to="wandb" \
  --seed=42
  # --train_text_encoder \
```

---
## Updates:

 - 25/9/7 fix the proble, of snr that tensor `base_weight` did't match `model_pred`'s shape.
 - add local multi_tag support, use it by: `--local_config_file_name` and  `--local_config_text_name`
 like
 <br>
 ```bash
  --local_config_text_name="text" \
  --output_dir="/content/drive/MyDrive/4_chars/output" \
 ```
 <br>
 - `protential problem:` since I changed the code realated to remote dataset load, so if you load datasets from huggingface, It might not be work, please use local.