from loguru import logger
from pathlib import Path
from data_juicer.config import init_configs
from data_juicer.core import Executor
import yaml
import json
from tqdm import tqdm
import os

def multi_data_process(data):
    print("------------开始将多轮问答数据转化为单轮问答数据-------------")
    # 将所有数据拆分为单轮问答
    result = []
    for i, conv in tqdm(enumerate(data)):
        for j in conv['conversations']:
            if j['from'] == 'human':
                result.append({})
                result[-1]['input'] = j['value']
                #result[-1]['input'] = ""
            else:
                result[-1]['output'] = j['value']
                result[-1]['ID'] = i
                result[-1]['category'] = conv['category'] if conv.get('category') else 'None'

    print("多轮问答数据样本量",len(data), "转换后单轮问答数据样本量",len(result))
    return result

@logger.catch(reraise=True)
def main():
    args = init_configs()
    config_file=args.config[0]
    with open(config_file, 'r') as file:
        config_ = yaml.safe_load(file)
    dataset_path = config_['dataset_path']

    if config_['data_type'] not in ['Alpaca', 'Sharegpt']:
        raise ValueError('data_type must be Alpaca or Sharegpt')

    #如果是多轮问答则处理数据为单轮数据
    if config_['data_type']=='Sharegpt':
        with open(config_['dataset_path'], 'r',encoding='utf-8') as file:
            multi_data = json.load(file)
        multi_length=len(multi_data)
        result=multi_data_process(multi_data)
        os.makedirs("./outputs/.cache/", exist_ok=True)
        with open("./outputs/.cache/preprocess_multi.jsonl", 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
        dataset_path="./outputs/.cache/preprocess_multi.jsonl"

    #如果是单轮问答则直接运行
    #数据筛选
    cfg_cmd = f'--config {config_file} --dataset_path {dataset_path}'.split()
    cfg = init_configs(args=cfg_cmd)
    cfg.open_tracer = True
    if cfg.executor_type == 'default':
        executor = Executor(cfg)
    elif cfg.executor_type == 'ray':
        from data_juicer.core.ray_executor import RayExecutor
        executor = RayExecutor(cfg)
    executor.run()
    trace_dir = executor.tracer.work_dir
    trace_files = list(Path(trace_dir).glob('*jsonl'))

    #单轮问答转换为多轮问答
    if config_['data_type']=='Sharegpt':
        delete_result=set()
        for i in tqdm(trace_files):
            with open(i, 'r',encoding='utf-8') as file:
                if 'duplicate' in str(i):
                    for line in file:
                        trace_data=json.loads(line)
                        if trace_data['dup1']['ID']!=trace_data['dup2']['ID']:
                            delete_result.add(trace_data['dup2']['ID'])
                else:
                    for line in file:
                        delete_result.add(json.loads(line)['ID'])

        left_num=list(set(range(multi_length))-delete_result)

        final_result=[]
        for j in left_num:
            final_result.append(multi_data[j])
        print("原始多轮问答数据样本量",multi_length, "筛选后多轮问答数据样本量",len(final_result))
        with open(config_['export_path'], 'w', encoding='utf-8') as file:
            json.dump(final_result, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
