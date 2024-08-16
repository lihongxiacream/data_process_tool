CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision bf16 \
    deita_pipeline/embed_datasets.py \
    --use_flash_attention true \
    --data_path "outputs/moss003_ifd_result.jsonl" \
    --output_path "outputs/.cache/output.pkl" \
    --batch_size_per_device 1 \
    --model_name_or_path "/luankexin/lihongxia/deita/src/deita/selection/embedder/Mistral-7B-v0.1"


CUDA_VISIBLE_DEVICES=0 python deita_pipeline/combined_filter.py \
    --data_path "outputs/moss003_ifd_result.jsonl" \
    --other_data_path "outputs/.cache/output.pkl" \
    --output_path "outputs/moss003_diversity.jsonl" \
    --threshold 0.6 \
    --data_size 50 \
    --sort_key "ifd_ppl"
