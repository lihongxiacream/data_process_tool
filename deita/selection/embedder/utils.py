import os
import torch
import random
import numpy as np
from datasets import Dataset
from typing import Sequence, Dict
from dataclasses import dataclass
from deita.selection.embedder.conversation import get_conv_template
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
IGNORE_INDEX=-100


def preprocess(
        samples: Dataset,
        conv_template,
        only_answer,
        max_length,
        tokenizer
    ) -> Dict:

        conv = get_conv_template(conv_template)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        sources = samples["conversations"]
        sample_ids = samples["input_idx"]

        # Apply prompt templates
        conversations = []

        if not only_answer:
            for i, source in enumerate(sources):
                # if roles[source[0]["from"]] != conv.roles[0]:
                #     # Skip the first one if it is not from human
                #     source = source[1:]

                conv.messages = []
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    # assert role == conv.roles[j % 2], f"{i}"
                    #assert role == conv.roles[j % 2], breakpoint()
                    conv.append_message(role, sentence["value"])
                conversations.append(conv.get_prompt())
        else:
            for i, source in enumerate(sources):
                if roles[source[0]["from"]] != conv.roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]

                messages = []
                for j, sentence in enumerate(source):
                    if j % 2 == 0:
                        continue
                    messages.append(sentence["value"])
                conversations.append("\n".join(messages))

        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids
        
        return dict(
            input_ids=input_ids,
            idx = sample_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
        
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, = tuple([instance[key] for instance in instances] for key in ("input_ids",))
        # input_ids = torch.tensor(input_ids)

        #bug fix
        def left_pad_sequence(sequences, batch_first=False, padding_value=0):
            max_len = max([s.size(0) for s in sequences])
            padded_sequences = []

            for seq in sequences:
                pad_size = max_len - seq.size(0)
                if pad_size > 0:
                    # 从左侧填充
                    padding = torch.full((pad_size,), padding_value, dtype=seq.dtype)
                    padded_seq = torch.cat([padding, seq], dim=0)
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)

            if batch_first:
                return torch.stack(padded_sequences)
            else:
                return torch.stack(padded_sequences).transpose(0, 1)

        input_ids = [torch.tensor(ids) for ids in input_ids]
        padded_ids = left_pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)


        input_ids = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(padded_ids), batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        instance_index = torch.tensor([instance["idx"] for instance in instances]).to(input_ids.device)

        return dict(
            idx = instance_index,
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def set_random_seed(seed: int):
    """
    Set the random seed for `random`, `numpy`, `torch`, `torch.cuda`.

    Parameters
    ------------
    seed : int
        The default seed.
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def batchlize(examples: list, batch_size: int, random_shuffle: bool, sort: bool, length_field = 'specific_length'):
    """
    Convert examples to a dataloader.

    Parameters
    ------------
    examples : list.
        Data list.
    batch_size : int.

    random_shuffle : bool
        If true, the dataloader shuffle the training data.
    
    sort: bool
        If true, data will be sort by its input length
    Returns
    ------------
    dataloader:
        Dataloader with batch generator.
    """
    size = 0
    dataloader = []
    length = len(examples)
    if (random_shuffle):
        random.shuffle(examples)
    
    new_examples = examples
    if sort:
        new_examples = sorted(examples, key = lambda x: len(x[length_field]))
    
    while size < length:
        if length - size > batch_size:
            dataloader.append(new_examples[size : size+batch_size])
            size += batch_size
        else:
            dataloader.append(new_examples[size : size+(length-size)])
            size += (length - size)
    return dataloader

def get_emb_name(**kwargs):
    
    if kwargs.get('model_path'):
        model_path = kwargs.pop("model_path")
        return os.path.basename(model_path)
    
    if kwargs.get('emb_name'):
        emb_name = kwargs.pop('emb_name')
        return os.path.basename(emb_name)