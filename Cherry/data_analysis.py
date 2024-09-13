import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from transformers import Qwen2ForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import time
from datasets import Dataset, DatasetDict
#from vllm import LLM, SamplingParams
#from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_loss_path", type=str, default='loss.json')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--filter_threash", type=float, default=1.2)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--prompt", type=str, default='qwen', help='qwen, alpaca')
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')
    args = parser.parse_args()
    return args

# log_softmax = nn.LogSoftmax(dim=-1)
# nll_loss = nn.NLLLoss(reduction='none')
loss_fn = nn.CrossEntropyLoss()

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
}

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(model, text):
    labels = torch.where(text['attention_mask'] == 1, text['input_ids'], -100).contiguous()

    #labels=input_ids.contiguous()
    #input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(text['input_ids'],labels=labels)
    #print(outputs.loss)

    #把logits和labels对齐
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 使用交叉熵损失函数
    losses=[]
    for i in range(outputs.logits.shape[0]):
        losses.append(loss_fn(shift_logits[i], shift_labels[i]).item())
    #print(losses)

    perplexity = torch.exp(torch.tensor(losses))
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(model, text):
    #给input部分进行label-100掩码
    labels = torch.where(text['attention_mask'] == 1, text['input_ids'], -100)
    # start_index = text.rfind(target_span)
    # start_token = len(tokenizer.encode(text[:start_index]))
    # end_token = input_ids.shape[1]
    #
    # labels = input_ids.clone()
    # labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(text['input_ids'], labels=labels)

    loss = outputs.loss
    #print(loss)

    #losses = []
    # logits = outputs.logits
    # for i in range(1, end_token):
    #     log_prob_dist = log_softmax(logits[0, i-1])
    #     true_token = input_ids[0, i]
    #     token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
    #     losses.append(token_loss.item())

    #把logits和labels对齐
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 使用交叉熵损失函数
    losses=[]
    for i in range(outputs.logits.shape[0]):
        #print(labels[i])
        losses.append(loss_fn(shift_logits[i], shift_labels[i]).item())
    #print(losses)

    perplexity = torch.exp(torch.tensor(losses))
    return perplexity.to('cpu'), losses


def main():
    args = parse_args()
    print(args)
    #sampling_params = SamplingParams(temperature=0.8, top_p=0.95,export_logits=True)
    # 加载配置和模型
    model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, cache_dir='../cache', output_hidden_states=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to('cuda')
    tokenizer =AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')
    model.eval()

    try:
        with open(args.data_path, "r") as f:
            data = json.load(f)
    except:
        data = []
        with open(args.data_path, "r") as f:
            for line in f:
                i=json.loads(line.strip())
                #i['output']=''.join(i['output'].split('\n')[1:])
                data.append(i)

    _data=[]
    for i in data:
        i['output'] = ''.join(i['output'].split('\n')[1:])
        _data.append(i)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(_data)
    sampled_data = _data[start_idx:end_idx]

    strat_time = time.time()
    input_data = {'input_ids':[]}
    output_data = {'input_ids':[],'attention_mask':[]}
    whole_data={'input_ids':[],'attention_mask':[]}

    #所有数据套上模板
    print('INFO|Start to tokenize data')
    response_token = tokenizer.encode('<|im_start|>assistant\n')
    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i['instruction']
        output_i = data_i['output']

        input_i = data_i['input'] if 'input' in data_i.keys() else ''
        if input_i == '':
            temp_dict = {'instruction':instruct_i}
            promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        else:
            temp_dict = {'instruction':instruct_i,'input':input_i}
            promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)

        input_tokens = tokenizer.encode(promt_to_use, truncation=True,max_length=args.max_length)
        output_tokens = tokenizer.encode(output_i, truncation=True,max_length=args.max_length)

        input_data['input_ids'].append(input_tokens)

        output_data['input_ids'].append(response_token + output_tokens)
        output_data['attention_mask'].append([0] * len(response_token) + [1] * len(output_tokens))

        whole_data['input_ids'].append(input_tokens + output_tokens)
        whole_data['attention_mask'].append([0] * len(input_tokens) + [1] * len(output_tokens))

    #创建datacollator便于padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # dataset=Dataset.from_dict(output_data)

    # print(data_collator([dataset[i] for i in range(len(dataset))]))
    #构建dataloader,输入tokenize后的数据
    dataloader_answer = DataLoader(Dataset.from_dict(output_data),collate_fn=data_collator,batch_size=args.batch_size, shuffle=False)
    dataloader_query = DataLoader(Dataset.from_dict(input_data),collate_fn=data_collator,batch_size=args.batch_size, shuffle=False)
    dataloader_whole = DataLoader(Dataset.from_dict(whole_data),collate_fn=data_collator,batch_size=args.batch_size, shuffle=False)

    #accelerator = Accelerator(num_processes=2,num_machines=2)
    #accelerator.wait_for_everyone()

    new_data=[]
    print('INFO|start to get ppl and loss')
    if args.mod == 'pre':
        if args.save_path[-3:] != '.pt':
            args.save_path += '.pt'

        #model, dataloader_query = accelerator.prepare(model, dataloader_query)
        for batch in tqdm(dataloader_query):

            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(model, batch.to('cuda'))
            for i in range(ppl_ins_alone.shape[0]):
                new_data.append({'ppl':[ppl_ins_alone[i],0,0],'sent_emb':[emb_ins_alone[i],0,0]})
            #print(new_data)

        torch.save(new_data, args.save_path)
        print('INFO|save the EMBEDDING in ' + args.save_path)

    elif args.mod == 'cherry':
        #model, dataloader_answer, dataloader_whole = accelerator.prepare(model, dataloader_answer,dataloader_whole)
        total_steps =len(dataloader_whole)
        for batch_an, batch_wh in tqdm(zip(dataloader_answer, dataloader_whole), total=total_steps):
            # instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length)
            # instruct_i_len = batch_wh['input_ids']  #instruct_i_input_ids.shape[1]

            ppl_out_alone, loss_list_alone = get_perplexity_and_embedding_part_text(model, batch_an.to('cuda'))
            ppl_out_condition, loss_list_condition = get_perplexity_and_embedding_part_text(model, batch_wh.to('cuda'))

            for i in range(ppl_out_alone.shape[0]):
                new_data.append({'ppl':[ppl_out_alone[i].item(),ppl_out_condition[i].item()],'token_loss':[loss_list_alone[i],loss_list_condition[i]]})

        with open(args.save_loss_path, "w") as file:
            json.dump(new_data,file)
        print('INFO|save the loss in '+args.save_loss_path)

        #计算ifd
        print('INFO|start to get ifd')
        assert len(sampled_data) == len(new_data)

        final_data=[]
        for i in tqdm(range(len(sampled_data))):
            json_data_i = sampled_data[i]
            loss_data_i = new_data[i]

            ppl_A_condition=loss_data_i['ppl'][1]
            ppl_A_direct=loss_data_i['ppl'][0]
            try:
                json_data_i['ifd_ppl'] = ppl_A_condition / ppl_A_direct
                if json_data_i['ifd_ppl']>args.filter_threash:
                    continue
            except ZeroDivisionError:
                json_data_i['ifd_ppl'] = 0
            final_data.append(json_data_i)


        final_data = sorted(final_data, key=lambda x: x['ifd_ppl'], reverse=True)

        sample_num = int(len(sampled_data) * args.sample_rate)

        if len(final_data) > sample_num:
            final_data = final_data[:sample_num]

        with open(args.save_path, "w") as file:
            json.dump(final_data,file, indent=4,ensure_ascii=False)

        final=[]
        for i in final_data:
            #i['output']='<|im_start|>assistant\n'+i['output']
            final.append(i)

        with open(args.save_path, "w") as file:
            json.dump(final,file, indent=4,ensure_ascii=False)

        print('INFO|selected data len:', len(final))

        print('INFO|save the ifd in '+args.save_path)

    print('INFO|Time Used:',(time.time()-strat_time)/60,'(min)')

if __name__ == "__main__":
    main()