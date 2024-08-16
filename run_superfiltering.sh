#ifd 筛选
CUDA_VISIBLE_DEVICES=0,1 python superfiltering/run_ifd.py \
--data_path demos/process_QA_data/data/CODE/code_alpaca_20k.json \
--model_name_or_path /luankexin/lihongxia/Superfiltering/gpt2_chinese \
--json_save_path outputs/code_result.jsonl \
--data_type Alpaca \
--ifd_rate 0.2

#single-turn diversity筛选
CUDA_VISIBLE_DEVICES=0,1 python superfiltering/run_diversity.py \
--data_path output.jsonl \
--embed_model_path /luankexin/lihongxia/Superfiltering/code_diversity_fla/all-MiniLM-L6-v2 \
--json_save_path outputs/diversity_result.jsonl \
--emb_type instruction \
--batch_size 64 \
--fla_num 3000