from openai import AzureOpenAI
from typing import List, Dict
import json
import tqdm
import copy

class GPT4Evaluation:
    def __init__(self,eval_data,character_profile,metric_prompt):
        self.eval_data = eval_data
        self.character_profile = character_profile
        self.metric_prompt = metric_prompt

        self.AZURE_ENDPOINT = "https://btree.openai.azure.com/"
        self.API_VERSION = "2024-05-01-preview"
        self.API_KEY = "2fc02fc43f394566ad206bea5a3dad34"
        self.DEPLOYMENT_NAME = "btree"

    def format_input(self,example):
        prompt = example['prompt'] if example.get('prompt') else ""
        input_text = "<RoleInfo>\n\n" \
            + str(self.character_profile[example['role']]) + "\n\n<Context>\n\n" + prompt +example['context'] + "\n\n<Response>\n\n" + example['model_output']
        return input_text

    def call_gpt4(self,messages: List[Dict], **kwargs):
        client = AzureOpenAI(
            azure_endpoint=self.AZURE_ENDPOINT,
            api_key=self.API_KEY,
            api_version=self.API_VERSION
        )
        response = client.chat.completions.create(
            model=self.DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.8 if 'temperature' not in kwargs else kwargs['temperature'],
            max_tokens=800 if 'max_tokens' not in kwargs else kwargs['max_tokens'],
            top_p=0.8 if 'top_p' not in kwargs else kwargs['top_p'],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None if 'stop' not in kwargs else kwargs['stop']
        )
        return response.choices[0].message.content

    # 交流能力评分：（模型在对话中表现出的语言流畅程度和逻辑连贯性）；
    # 角色一致性评分：（模型生成的对话是否符合角色设定的性格、背景和行为特征）；
    # 角色扮演能力评分：（模型在模拟角色时的逼真程度和情感表达能力）。
    def process_data(self):
        result={}
        for data in tqdm.tqdm(self.eval_data):
            input_text = self.format_input(data)
            self.metric_prompt = self.metric_prompt if self.metric_prompt else """交流能力评分：（模型在对话中表现出的语言流畅程度和逻辑连贯性）；
            角色一致性评分：（模型生成的对话是否符合角色设定的性格、背景和行为特征）；
            角色扮演能力评分：（模型在模拟角色时的逼真程度和情感表达能力）。"""
            message= [
            {"role": "system",
             "content": """
            评估角色扮演大模型的质量并打分：
            1. 基于user输入的角色设定数据、历史人物对话数据和角色扮演模型生成数据，评估模型的交流能力、角色一致性和角色扮演能力
            2. 基于如下指标打分，不需要输出理由:{}
            3. 使用五分制进行评分，其中5分表示非常优秀，1分表示非常不足，分数为float32类型,如1.98046875
            4. 以 JSON 格式返回评分结果，其中字典中的键是打分指标，值是评分
            """.format(self.metric_prompt)},
            {"role": "user", "content": input_text}
            ]
            #data['score']=self.call_gpt4(message)
            result[data['id']] = self.call_gpt4(message)

        f = open('results/evaluation.jsonl','w', encoding='utf-8')
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

        return result
            #print(data['id'],call_gpt4(message))