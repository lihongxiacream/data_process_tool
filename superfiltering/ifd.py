import os
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModel
import math

class Superfiltering_IFD():
    def __init__(self, data_path,post_data_path,model_name_or_path,sample_rate,json_save_path,data_type):

        self.data_path = data_path
        self.model_name_or_path = model_name_or_path
        self.sample_rate = sample_rate #每组数据要多少比例
        self.json_save_path = json_save_path
        self.post_data_path=post_data_path
        self.data_type=data_type
        self.max_length=1024
        self.key_name = 'ifd_ppl'
        self.filter_threash = 1

        self.PROMPT_DICT_NONE = {
        "prompt_input": (
            "{instruction}\n{input}\n"
        ),
        "prompt_no_input": (
            "{instruction}\n"
        )}

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    # Used to get the ppl and emb for the whole input
    def get_perplexity_and_embedding_whole_text(self,tokenizer, model, text, max_length):

        try:
            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids.contiguous())
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.to('cpu').item(), loss.to('cpu').item()

        except:
            return 0, 0

    # Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
    def get_perplexity_and_embedding_part_text(self,tokenizer, model, text, target_span, max_length):

        try:
            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

            start_index = text.rfind(target_span)
            start_token = len(tokenizer.encode(text[:start_index]))
            end_token = input_ids.shape[1]

            labels = input_ids.clone()
            labels[0, :start_token] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=labels)

            loss = outputs.loss
            perplexity = torch.exp(loss)

            return perplexity.to('cpu').item(), loss.to('cpu').item()

        except:
            return 0, 0

    #对于单轮问答数据进行预处理
    def preprocess_single(self,data):
        result=[]
        for i in data:
            if i.get("instruction","")=="":
                if i.get('input',"")=="":
                    raise ValueError("instruction和input不能同时为空")
                i['instruction']=i['input']
                i['input']=""
            result.append(i)
        return result

    #对于多轮问答数据进行预处理，整理成单轮问答格式计算ppl
    def preprocess_multi(self,data):

        print("------------开始将多轮问答数据转化为单轮问答数据-------------")
        # 将所有数据拆分为单轮问答
        result = []
        for i, conv in tqdm(enumerate(data)):
            for j in conv['conversations']:
                if j['from'] == 'human':
                    result.append({})
                    result[-1]['instruction'] = j['value']
                    result[-1]['input'] = ""
                else:
                    result[-1]['output'] = j['value']
                    result[-1]['ID'] = i

        print(len(data),len(result))
        # with open("./process_data.jsonl", 'w') as file:
        #     json.dump(result, file, indent=4, ensure_ascii=False)
        return result

    #筛选多轮问答数据
    def select_data_multi(self,json_data,old_data):

        print("------------开始基于IFD值筛选多轮问答数据-------------")
        #计算一轮问答均分
        #result_conv = []
        conv = -1
        for i, instruct in enumerate(json_data):

            # input = instruct['instruction']
            # output = instruct['output']

            if instruct['ID'] > conv:  # 新的一轮对话更新所有信息
                if conv>=0:
                    old_data[instruct['ID']-1]['ifd_ppl'] = sum(ppl) / len(ppl)
                ppl = []
                conv = instruct['ID']

            ppl.append(instruct['ifd_ppl'])
            # result_conv[-1]['conversations'].append({"from": "human", "value": input})
            # result_conv[-1]['conversations'].append({"from": "gpt", "value": output})
        old_data[instruct['ID']]['ifd_ppl'] = sum(ppl) / len(ppl)
        # with open("outputs/.cache/test.jsonl",'w') as file:
        #     json.dump(old_data,file,indent=4,ensure_ascii=False)
        def sort_key(x):
            # Check if the value is nan
            if math.isnan(x['ifd_ppl']):
                return (0, 0)
            return (1, x['ifd_ppl'])

        sample_num = int(len(old_data) * self.sample_rate)
        filtered_data = [x for x in old_data if
                     (isinstance(x['ifd_ppl'], (int, float)) and x['ifd_ppl'] < self.filter_threash)]
        new_data = sorted(filtered_data, key=sort_key, reverse=True)
        if len(new_data) > sample_num:
            new_data = new_data[:sample_num]
        return new_data

    #筛选单轮问答数据
    def select_data_single(self,json_data):

        print("----------开始IFD值筛选单轮问答数据-------------")

        def sort_key(x):
            # Check if the value is nan
            if math.isnan(x[self.key_name]):
                return (0, 0)
            return (1, x[self.key_name])

        filtered_data = [x for x in json_data if
                         (isinstance(x[self.key_name], (int, float)) and x[self.key_name] < self.filter_threash)]
        new_data = sorted(filtered_data, key=sort_key, reverse=True)

        sample_num = int(len(json_data) * self.sample_rate)

        if len(new_data) > sample_num:
            new_data = new_data[:sample_num]

        return new_data

    def run(self):

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir='../cache')
        model.eval()
        # model_en = AutoModelForCausalLM.from_pretrained("/luankexin/lihongxia/Superfiltering/gpt2", device_map="auto", cache_dir='../cache', output_hidden_states=True)
        # tokenizer_en = AutoTokenizer.from_pretrained("/luankexin/lihongxia/Superfiltering/gpt2", cache_dir='../cache')
        # model_en.eval()

        print("------------模型加载结束--------------")
        #判断一下输入是否是列表 不是列表的话按行读入
        try:
            with open(self.data_path, "r") as f:
                read_data = json.load(f)
        except:
            #if not isinstance(data, list):
            read_data=[]
            with open(self.data_path, "r") as f:
                for line in f:
                    read_data.append(json.loads(line.strip()))


        #判断是否是多轮问答数据，是的话预处理成单轮问答
        if self.data_type=="Sharegpt":
            data=self.preprocess_multi(read_data)
        else:
            data=self.preprocess_single(read_data)

        # start_idx = args.start_idx
        # end_idx = self.end_idx if self.end_idx != -1 else len(data)
        # data = data[:end_idx]

        # if not os.path.exists(save_path):
        #每次重新更新一下debug文件
        save_path='outputs/.cache/debug.jsonl'
        os.makedirs("./outputs/.cache/", exist_ok=True)
        with open(save_path, "w") as file:
            pass  # Creates an empty file

        #若跑到一半终止可以从此时继续跑
        # with open(save_path, "r") as file:
        #     exsisting_num =  sum(1 for _ in file)
        # data = data[exsisting_num:]

        print("----------数据读取完成----------")

        prompt_no_input = self.PROMPT_DICT_NONE["prompt_no_input"]
        prompt_input = self.PROMPT_DICT_NONE["prompt_input"]

        print("-------开始计算指令的IFD值----------")
        pt_data = []
        for i in tqdm(range(len(data))):

            data_i = data[i]
            instruct_i = data_i['instruction']
            output_i = data_i['output']

            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = prompt_no_input.format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = prompt_input.format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

            #判断当前instruct是中文还是英文，选择不同的模型
            # lang = detect(instruct_i)
            # if lang == 'zh-cn':
            #     model=model_zh
            #     tokenizer=tokenizer_zh
            # else:
            #     model=model_en
            #     tokenizer=tokenizer_en

            instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            instruct_i_len = instruct_i_input_ids.shape[1]

            if output_i == '':
                temp_data_i = {}
            else:
                ppl_out_alone, loss_out_alone = self.get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, self.max_length-instruct_i_len+1)
                ppl_out_condition, loss_out_condition = self.get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, self.max_length)

                temp_data_i = {}
                temp_data_i['ppl'] = [0,ppl_out_alone,0,ppl_out_condition]
                temp_data_i['loss'] = [0,loss_out_alone,0,loss_out_condition]

            pt_data.append(temp_data_i)
            with open(save_path, "a") as file:
                file.write(json.dumps(temp_data_i) + '\n')

        print("PPL computation finished",save_path)
        print("-------IFD值插入数据----------")

        assert len(data) == len(pt_data)
        #put analysis to data
        new_data = []
        for i in tqdm(range(len(pt_data))):

            json_data_i =data[i]

            pt_data_i = pt_data[i]
            if pt_data_i == {}:
                ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = np.nan, np.nan, np.nan, np.nan
                loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = np.nan, np.nan, np.nan, np.nan
            else:
                ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = \
                    pt_data_i['ppl'][0], pt_data_i['ppl'][1], pt_data_i['ppl'][2], pt_data_i['ppl'][3]
                loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = \
                    pt_data_i['loss'][0], pt_data_i['loss'][1], pt_data_i['loss'][2], pt_data_i['loss'][3]

            json_data_i['ppl_A_direct'] = ppl_A_direct
            json_data_i['ppl_A_condition'] = ppl_A_condition
            try:
                json_data_i['ifd_ppl'] = ppl_A_condition / ppl_A_direct
            except ZeroDivisionError:
                json_data_i['ifd_ppl'] = 0

            new_data.append(json_data_i)

        # 提取目录路径
        directory = os.path.dirname(self.post_data_path)

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        #记录带有IFD值的数据
        with open(self.post_data_path, "w") as file:
            file.write(json.dumps(new_data))

        print("IFD computation finished", self.post_data_path)

        if self.data_type=='Sharegpt':
            final_result=self.select_data_multi(new_data,read_data)
        else:
            final_result=self.select_data_single(new_data)

        # 提取目录路径
        dir_ = os.path.dirname(self.json_save_path)

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        print("原始数据量",len(read_data),"筛选后数据量",len(final_result))
        with open(self.json_save_path, 'w') as file:
            json.dump(final_result,file, ensure_ascii=False)

        print('Done: Data Selection:', self.json_save_path)

    def run_only_select(self):
        #带有IFD值的数据读取
        with open(self.post_data_path, "r") as f:
            new_data = json.load(f)
        print("————————————数据读取完毕————————————")

        if self.data_type=='Sharegpt':
            final_result=self.select_data_multi(new_data,new_data)
        else:
            final_result=self.select_data_single(new_data)

        print(len(final_result))
        with open(self.json_save_path, 'w') as file:
            json.dump(final_result,file, ensure_ascii=False)

        print('Done: Data Selection:', self.json_save_path)


