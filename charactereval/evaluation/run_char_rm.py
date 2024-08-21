import sys
import torch
import json
import copy
import tqdm
from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer
import os

class CharacterRM:
    def __init__(self):
        self.device = torch.device('cuda')
        self.max_seq_length = 4096
        self.reward_model_path = 'BaichuanCharRM/'
        self.tokenizer = BaichuanTokenizer.from_pretrained(self.reward_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.base_model = BaichuanCharRM.from_pretrained(self.reward_model_path, torch_dtype=torch.bfloat16).to(self.device)

    def format_input(self,example,character_profile):
        prompt= example['prompt'] if example.get('prompt') else ""
        input_text = "<RoleInfo>\n\n" \
                     + str(character_profile[example['role']]) + "\n\n<Context>\n\n" + prompt + example[
                         'context'] + "\n\n<Response>\n\n" + example['model_output'] + "\n\n<Dimension>\n\n" + example[
                         "metric_zh"]
        return input_text

    def evaluate(self,datas,character_profile):

        with open('data/metric.jsonl','r', encoding='utf-8') as f:
            metric = json.load(f)

        # with open(eval_path,'r', encoding='utf-8') as f:
        #     datas = json.load(f)
        #
        # with open(character_path, "r", encoding='utf-8') as f:
        #     character_profile = json.load(f)

        torch.cuda.empty_cache()#清理缓存
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

        #数据格式转化
        records = []

        for data in datas:
            if data['model_output'] is not None and data['model_output'] != "ERROR":
                model_output = data['model_output'].split("\n")[0] # Prevent continuous generation
                data['model_output'] = model_output
                for x in metric["metric"]:
                    data['metric_zh']= x[1]
                    tmp = copy.deepcopy(data)
                    records.append(tmp)
        #records=copy.deepcopy(records[:100])#注意

        #baichuanRM评估
        result={}
        for record in tqdm.tqdm(records):
            input_text = self.format_input(record,character_profile)
            input_ids = self.tokenizer.encode(text=input_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[-self.max_seq_length:]
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

            with torch.no_grad():
                score = self.base_model(input_ids=input_ids)[1].item() * 4 + 1
                result[record['id']]=result[record['id']] if result.get(record['id']) else {}
                result[record['id']][record['metric_zh']]=score
                #record['score'] = score

        f = open('results/evaluation.jsonl','w', encoding='utf-8')
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
        return result

#a=CharacterRM()
#a.evaluate(f'data/eval_data.jsonl',f'data/character_profiles.json')