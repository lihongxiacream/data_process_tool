from tqdm import tqdm
import numpy as np
import time
import json
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import math
from submodlib.functions.facilityLocation import FacilityLocationFunction

class Superfiltering_Diversity():
    def __init__(self, data_path,embed_model_path,emb_type,batch_size,json_save_path,diversity):

        self.data_path = data_path
        self.embed_model_path = embed_model_path
        self.emb_type = emb_type
        self.json_save_path = json_save_path
        self.batch_size = batch_size
        self.fla_num = diversity

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    #求解facility location
    def do_fla(self,X, number_all):
        start_time = time.time()

        Y = X
        obj = FacilityLocationFunction(n=number_all, mode="dense", data=Y, metric="euclidean")
        greedyList = obj.maximize(budget=self.fla_num, optimizer='LazyGreedy', stopIfZeroGain=False,
                                  stopIfNegativeGain=False, verbose=False)
        idx_list = [tuple_i[0] for tuple_i in greedyList]

        print('FLA time used:', (time.time() - start_time) / 60, '(min)')
        return idx_list

    #生成facility location文本输入
    def combine_sentences(self,filtered_data):
        sent_all = []
        for dict_i in filtered_data:

            if 'input' in dict_i.keys():
                instruction_i = dict_i['instruction'] + '\n' + dict_i['input'] + '\n'
            else:
                instruction_i = dict_i['instruction'] + '\n'

            if self.emb_type == 'whole':
                sent_all.append(instruction_i + dict_i['output'])
            elif self.emb_type == 'instruction':
                sent_all.append(instruction_i)
            elif self.emb_type == 'response':
                sent_all.append(dict_i['output'])

        return sent_all

    #生成embedding
    def get_sentence_embeddings(self,sentences):

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),min=1e-9)

        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.embed_model_path)
        model = AutoModel.from_pretrained(self.embed_model_path).to(self.device)
        all_sentence_embeddings = []

        for i in tqdm(range(0, len(sentences), self.batch_size)):
            # Process sentences in batches
            batch_sentences = sentences[i:i + self.batch_size]

            # Tokenize sentences and convert to input format expected by the model
            encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

            # Move encoded input to the same device as the model
            encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

            # Get model's output (without any specific head)
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_sentence_embeddings.append(sentence_embeddings)

        # Concatenate all embeddings from each batch
        all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0)

        return all_sentence_embeddings.cpu()

    #facility location对单轮问答数据进行筛选
    def diversity(self,filtered_data):

        if self.fla_num > len(filtered_data):
            raise ValueError("The number of selected data is more than the original data")
        # get the embedding
        sentences = self.combine_sentences(filtered_data)
        print(len(sentences))
        embeddings = self.get_sentence_embeddings(sentences)
        print(embeddings.shape)
        # do fla
        X = embeddings.numpy()
        fla_idxs = self.do_fla(X, len(sentences))

        final_json_data_ori = [filtered_data[i] for i in fla_idxs]
        return final_json_data_ori

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

    def run(self):

        # 判断一下输入是否是列表 不是列表的话按行读入
        try:
            with open(self.data_path, "r") as f:
                read_data = json.load(f)
        except:
            read_data = []
            with open(self.data_path, "r") as f:
                for line in f:
                    read_data.append(json.loads(line.strip()))

        print("原始数据量", len(read_data))
        data = self.preprocess_single(read_data)
        final_result = self.diversity(data)
        print("筛选后数据量",len(final_result))

        with open(self.json_save_path, 'w') as file:
            json.dump(final_result, file, indent=4,ensure_ascii=False)

        print('Done: Data Selection:', self.json_save_path)