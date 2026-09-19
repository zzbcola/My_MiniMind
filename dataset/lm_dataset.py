from torch.utils.data import Dataset
import torch
import json
import random
from datasets import load_dataset, Features, Value

from trainer_utils import Logger


def pre_processing_chat(conversations, add_system_ratio=0.2):
    '''
    在预训练阶段注入的system prompt
    :param conversations:输入的对话
    :param add_system_ratio:
    :return: 注入system prompt后的完整的对话
    '''

    # 对于tool use 数据完整保留不做注入system prompt的处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    '''
    将数据中的的思考部分以1-empty_think_ratio的概率移除
    '''

    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    '''
    初始化预训练的数据集处理
    '''
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        '返回数据长度'
        return len(self.samples)

    def __getitem__(self, index):
        '通过index索引获取数据的方法，具体为实例化之后加上[num]调用'
        sample = self.samples[index]
        # 将text部分进行tokenizer
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 在text前后部分加上“开始符号”和“结束符号”
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        return tokens

    def collate_fn(self, batch):
        # 每个 batch 只填充到本批次最长样本，减少固定 max_length 带来的无效计算
        batch_max_length = min(max(len(tokens) for tokens in batch), self.max_length)
        input_ids_list = []
        labels_list = []
        for tokens in batch:
            input_ids = tokens[:batch_max_length]
            input_ids = input_ids + [self.tokenizer.eos_token_id] * (batch_max_length - len(input_ids))
            labels = input_ids.copy()
            if len(tokens) < batch_max_length:
                labels[len(tokens):] = [-100] * (batch_max_length - len(tokens))
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        return torch.tensor(input_ids_list, dtype=torch.long), torch.tensor(labels_list, dtype=torch.long)

class SFTDataset(Dataset):
    '''
    SFT的数据类型简单如下：
    {
      "conversations": [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是..."}
      ]
    }
    也有包含resonning_content, tools, too_calls的数据内容，详情可以查看数据文件

    改类主要处理sft的数据
    最后返回input_ids 和 labels的tensor，结构为self.bos_id + token + self.ens_id (+ self.tokenizer.pad_token_id), 长度为max_length

    '''
    def __init__(self, jsonl_path: str, tokenizer, max_length:int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 设置好数据的形式
        # 由于sft中存在可选字段"reasoning_content", "tools", "tool_calls"等，需要构造feature
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'),
                                               'reasoning_content': Value('string'), 'tools': Value('string'),
                                               'tool_calls': Value('string')
                                               }]
                            })
        self.samples = load_dataset('json', data_files = jsonl_path, split = 'train', features = features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens = False).input_ids # 设置开始符号
        self.ens_id = tokenizer(f'{tokenizer.ens_token}\n', add_special_tokens = False).input_ids # 这是结束符号

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        '''
        对sft数据集中的tools和tool_calls处理为列表形式，并且对这些数据进行完整的tokenizer进行返回
        '''
        messages = []
        tools = None
        # 对message中的tools（函数/工具列表描述等）和tool_calls（模型决定调用的工具）两种类型做处理，将原始的json字符转化为列表
        for message in conversations:
            message = dict(message)
            if message.get('role') == 'system' and message.get('tools'):
                tools = json.loads(message['tools']) if isinstance(message['tools'], str) else message['tools']
            if message.get('tool_calls') and isinstance(message['tool_calls'], str):
                message['tool_calls'] = json.loads(message['tool_calls'])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenizer = False,
            add_generation_prompt = False,
            tools = tools
        )

    def generate_labels(self, input_ids) -> list:
        '''
        找出input_ids中self.bos_id, real_token, self.ens_id三部分，将除了real_token部分保留tokenizer编码之外，
        其他都设置成-100作为labels列表，方便只对模型回答的部分token进行loss计算
        修改过程如下：
        input_ids:
        [user token] [user token] [assistant标记] [答案token] [答案token] [结束标记] [pad]

        labels:
        [-100]       [-100]       [-100]        [答案token] [答案token] [结束标记] [-100]
        '''
        labels = [-100] * len(input_ids) # 全部初始化为-100，表示全部忽略不计算loss
        i = 0
        # 寻找开始符号到结束符号之间的有效token索引，用start和end记录
        while i < len(input_ids):
            # input_ids[i:i+len(self.bos_id)]表示类似与<>start等字符，如果判断成立说明i开始是为起始符，需要将i + len(self.bos_id)的长度开始设置为回答token的开头
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 结束符号判断原因与启始符号设置原因类似
                    if input_ids[end: end + len(self.ens_id)] == self.ens_id:
                        break
                    end += 1
                for j in range(start, min(end + self.ens_id, self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.ens_id) if end + len(self.ens_id) < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        '''
        通过索引获取sft数据
        '''
        sample = self.samples[index]
        conversation = pre_processing_chat(sample) # 概率注入system_prompt
        prompt = self.create_chat_prompt(conversation) # 处理tools和tool_calls后进行tokenizer
        prompt = post_processing_chat(prompt) # 概率删除think标签
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] # 截取设定的最长长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) #将不足self.max_length部分进行填充，保证每个batch长度一直
        labels = self.generate_labels(input_ids)

        return torch.tnesor(input_ids, type=torch.long), torch.tnesor(labels, type=torch.long)


class DPODataset(Dataset):
    '''

    数据内容简单如下所示：
    {
      "chosen": [
        {"role": "user", "content": "介绍机器学习"},
        {"role": "assistant", "content": "机器学习是一种..."}
      ],
      "rejected": [
        {"role": "user", "content": "介绍机器学习"},
        {"role": "assistant", "content": "机器学习就是让机器变聪明。"}
      ]
    }

    '''
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        # 返回数据集的长度
        return len(self.samples)

    def generate_loss_mask(self, input_ids):
        '''
        dpo版本的labels构建，与sft不同，dpo只需要记录0和1，对于self.bos_id到self.eos_id部分，设置为1其他设置为0
        例子如下：
        loss_mask:
        [0, 0, 0, 1, 1, 1, 0, 0]
        dpo可以使用0和1作为labels的原因是 DPO 不直接调用模型内部的交叉熵，而是自己计算每个 token 的 log probability。
        '''
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(loss_mask):
            if input_ids[i : i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenizer=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apple_chat_template(
            rejected, tokenizer=False, add_generation_prompt=False
        )

        # 对拒绝prompt部分进行概率移除思考
        rejected_prompt = post_processing_chat(rejected_prompt)

        # token编码，编码为[batch_size, max_length]的固定长度
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 获取两者的input_ids的部分，并且对回答部分设置01mask
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # 手动构造next_token prediction，通过错位方面后续预测计算loss
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

# class RLAIFDataset(Dataset):
#     def __init__(self, json_path, tokenizer, max_length=1024, thinking_ratio=0.5):

