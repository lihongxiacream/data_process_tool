pip install -r requirements.txt

cd lihongxia/Cherry_LLM
source cherry_env/bin/activate

#选择Pre-Experienced数据集
#计算embedding
CUDA_VISIBLE_DEVICES=0 python Cherry/data_analysis.py \
    --data_path data/process2.jsonl \
    --save_path result/cherry_data_pre.pt \
    --model_name_or_path /luankexin/leirongzhen/train_llm/LLaMA-Factory/Qwen2-7B-Instruct \
    --batch_size 4 \
    --max_length 8192 \
    --mod pre

#根据聚类结果筛选1000条训练数据
CUDA_VISIBLE_DEVICES=0 python Cherry/data_by_cluster.py \
    --pt_data_path result/cherry_data_pre.pt \
    --json_data_path  data/process2.jsonl \
    --json_save_path result/cherry_train_data.json \
    --sample_num 10 \
    --kmeans_num_clusters 100 \
    --low_th 25 \
    --up_th 75

#计算ifd并筛选cherry数据
CUDA_VISIBLE_DEVICES=1 python Cherry/data_analysis.py \
    --data_path data/process_new.json \
    --save_path Cherry/result/cherry_ifd2.json \
    --sample_rate 0.3 \
    --batch_size 4 \
    --model_name_or_path /luankexin/lihongxia/LLAMA_Factory/models/qwen2_7B_lora_sft_cherry_pre \
    --max_length 8192 \
    --mod cherry