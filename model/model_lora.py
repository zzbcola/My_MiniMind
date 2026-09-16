import torch
from torch import nn, optim

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank =  rank
        # 设置两个线形层作为lora矩阵
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 对矩阵A做高斯分布初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 对矩阵B进行全零初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    '''
    将lora应用到模型上面
    '''
    for name, module in model.named_modules():
        '''
        遍历所有模型层
        对于所有输入和输出相同的线性层都设置为lora类
        最后动态注入lora层
        '''
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank).to(model.device)
            setattr(module, "lora", lora) # 设置一个新的lora模块
            original_forward = module.forward
            # 注入lora的forward，与原模块的forward相加
            # 这里需要显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora

def load_lora(model, path):
    '''
    替换lora模块的名称，方便后续加载
    '''
    state_dict = torch.load(path, map_location=model.device)
    # 将所有module.开头的模块中的kv对都存储到state_dict
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    # 对于state_dict对中{name}.lora替换成空白
    '''
    替换例子如下：
    {
    'transformer.h.0.attn.q_proj.lora.A.weight': tensor(...),  
    'transformer.h.0.attn.q_proj.lora.B.weight': tensor(...),
    }
       
    替换为：
    
    {
    'A.weight': tensor(...),
    'B.weight': tensor(...),
    }
    '''
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {k.replace(f'{name}.lora.', ''):v for k, v in state_dict.items() if f'{name}.lora' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    '''
    保存lora层的所有k v对到本地
    '''
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    # 获取所有lora层的k，v保存
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startwith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    # 获取模型中除去lora模块外的所有模块k和对应张量v
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    # 遍历所有模块，对里面模块的权重进行修改
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and  '.lora.' not in name:
            # 非lora权重不变
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu()
            # lora模块需要融合A和B权重作为新权重
        if hasattr(module, 'lora'):
            state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
