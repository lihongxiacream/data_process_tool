# 支持的数据格式
## 1. Alpaca
```
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）"
  }
]
```
## 2. Sharegpt
```
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）"
  }
]
```
# 环境搭建
## 1. pip 安装
```
cd <path_to_data_juicer>
pip install -v -e .
#若出现numpy不兼容问题则conda install numpy
```
若需使用Single-turn Diversity指标还需如下命令
```
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
```
## 2. docker安装
`docker build -t datajuicer/data-juicer:latest .`
## 3. A800虚拟环境激活
```
cd luankexin/lihongxia/data-juicer
conda activate juicer_env
```
# 数据清洗前准备
若使用A800可忽略
1. 若需使用IFD筛选，从huggingface下载gpt2和gpt2-chinese并部署
2. 若需使用single-turn Diversity筛选，从huggingface下载all-MiniLM-L6-v2并部署
3. 若需使用multi-turn Diversity筛选，从huggingface下载Mistral-7B-v0.1并部署
# 数据筛选命令
## 1. Data juicer
## 1.1 启动命令
` python tools/process_data.py --config configs/demo/process_zh.yaml`
- 注意：使用未保存在本地的第三方模型或资源的算子第一次运行可能会很慢，因为这些算子需要将相应的资源下载到缓存目录中。默认的下载缓存目录为~/.cache/data_juicer。
- 用户可以自己构建如下的配置文件，也可以使用已经配置好的数据清洗配置文件，文件路径为configs/demo/process.yaml，configs/demo/process_zh.yaml和configs/demo/process_en.yaml，注意修改dataset_path和text_keys
## 1.2 参数配置
如下为yaml文件为例，具体的算子介绍请看https://github.com/modelscope/data-juicer/blob/main/docs/Operators_ZH.md
```
project_name: 'demo-process' #项目名称
dataset_path: './demos/data/moss003_qwen.jsonl' #原始数据集路径
np: 4  #处理数据集的子进程数
text_keys: 'input' #清洗操作对应的字段名（若是多轮问答，则input和output分别对应user content和assistant content）
data_type: 'Sharegpt' #数据格式(填Sharegpt或Alpaca)
export_path: './outputs/test/moss003-processed.jsonl'#输出数据路径

#算子列表
process:
  - language_id_score_filter: #算子名称
      lang: zh #算子参数设置
      
  - document_deduplicator: 
    lowercase: true 
    ignore_non_character: true
```
## 2. IFD_gpt2
```
CUDA_VISIBLE_DEVICES=0,1 python superfiltering/run_ifd.py \
--data_path outputs/test/moss003-processed.jsonl \   
--model_name_or_path /luankexin/lihongxia/Superfiltering/gpt2_chinese \
--json_save_path outputs/multi_result.jsonl \
--data_type Sharegpt \
--ifd_rate 0.2 \
```
- 命令行参数：\
  --model_name_or_path  计算IFD模型路径 \
  --data_type 数据格式，只支持Sharegpt和Alpaca两种 \
  --ifd_rate  IFD筛选比例（例：筛选IFD值排名前20%数据） 
- 注意：如果只评估数据的ifd值，不筛选数据可以把ifd_rate设为1，但最后输出的文本会过滤掉IFD大于1的样本。因为如果 IFD 分数大于 1，则基于指令的loss值甚至大于直接生成的loss值，这意味着给定的指令没有为预测响应提供有用的上下文。在这种情况下，我们认为指令和相应的响应之间存在不一致。因此，我们选择过滤这些可能错位的样本。
- 由于使用gpt2预训练模型生成结果，因此支持的文本最大长度为1024，若过长直接truncated，若需要支持更长的文本，可使用下面的IFD_qwen7B方法筛选数据
- 若ppl和loss的计算终止，可查看outputs/.cache/debug.jsonl查看已经计算的部分值
## 3. IFD_qwen7B
主要筛选流程为：聚类筛选少量多样数据；拿这部分数据对模型进行初步训练；初学模型计算原始数据的IFD指标，并筛选
### 3.1 选择Pre-Experienced数据集
#### 3.1.1 计算embedding
```
CUDA_VISIBLE_DEVICES=0 python Cherry/data_analysis.py \
    --data_path data/process2.jsonl \
    --save_path result/cherry_data_pre.pt \
    --model_name_or_path /luankexin/leirongzhen/train_llm/LLaMA-Factory/Qwen2-7B-Instruct \
    --batch_size 4 \
    --max_length 8192 \
    --mod pre
```
--mod pre  选择Pre-Experienced模式 cherry  筛选ifd数据模式
#### 3.1.2 根据聚类结果筛选训练数据
```
CUDA_VISIBLE_DEVICES=0 python Cherry/data_by_cluster.py \
    --pt_data_path result/cherry_data_pre.pt \
    --json_data_path  data/process2.jsonl \
    --json_save_path result/cherry_train_data.json \
    --sample_num 10 \
    --kmeans_num_clusters 100 \
    --low_th 25 \
    --up_th 75
```
--pt_data_path  步骤3.1.1embedding路径 \
--json_data_path  原始数据集路径
### 3.2 计算ifd并筛选cherry数据
```
CUDA_VISIBLE_DEVICES=1 python Cherry/data_analysis.py \
    --data_path data/process2.json \
    --save_path Cherry/result/cherry_ifd2.json \
    --sample_rate 0.3 \
    --batch_size 4 \
    --model_name_or_path /luankexin/lihongxia/LLAMA_Factory/models/qwen2_7B_lora_sft_cherry_pre \
    --max_length 8192 \
    --mod cherry
```
## 4. Single-turn Diversity
```
CUDA_VISIBLE_DEVICES=0,1 python superfiltering/run_diversity.py \
--data_path data_process/output.json \
--embed_model_path /luankexin/lihongxia/Superfiltering/code_diversity_fla/all-MiniLM-L6-v2 \  #计算embedding模型路径
--json_save_path outputs/multi_result.jsonl \
--emb_type instruction \
--batch_size 32 \  
--fla_num 3000
```
--emb_type  多样性筛选针对的字段，仅支持whole,instruction,response \
--fla_num  预计需要利用diversity的筛选数量
## 5. Multi-turn Diversity
由于deita会根据预先的IFD排序数据（非筛选数据）进行多样性的筛选，即基于IFD值由高到低抽取数据（保证抽取多样性数据的高质量），因此输入数据必须有一个sort_key
### 5.1 计算embedding
```
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision bf16 \
    deita_pipeline/embed_datasets.py \
    --use_flash_attention true \
    --data_path "outputs/moss003_ifd_result.jsonl" \
    --output_path "outputs/.cache/output.pkl" \
    --batch_size_per_device 1 \
    --model_name_or_path "/luankexin/lihongxia/deita/src/deita/selection/embedder/Mistral-7B-v0.1"
```
### 5.2 基于diversity筛选样本
```
CUDA_VISIBLE_DEVICES=0 python deita_pipeline/combined_filter.py \
    --data_path "outputs/moss003_ifd_result.jsonl" \
    --other_data_path "outputs/.cache/output.pkl" \
    --output_path "outputs/moss003_diversity.jsonl" \
    --threshold 0.7 \
    --data_size 50 \
    --sort_key "ifd_ppl"
```
--other_data_path  embedding文件路径 \
--threshold  embedding距离阈值（必须小于1），越小代表距离越远（两个文本的相似度越小） \
--data_size  需要筛选出的数据量 \
--sort_key  排序分数的键