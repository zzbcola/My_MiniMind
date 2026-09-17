import math, torch, torch.nn.functional as F

from click.termui import hidden_prompt_func
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

class RMSNorm(nn.Module):
    '''
    归一化类
    '''
    def __init__(self, dim: int, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 维度为dim的全是1向量，作为参数，通过学习进行调整

    def norm(self, x):
        '''
        归一化操作：乘以平方均值的倒数，使得里面是数值接近1，在1的附近，只关心它们的“能量大小”
        '''
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        '''
        归一化后乘上参数权重，用于训练传播调整
        '''
        return (self.weight * self.norm(x.float())).type_as(x)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    '''
    旋转位置编码
    '''
    def rotate_half(x):
        '''
        将向量逆时针旋转90度
        '''
        return torch.cat((-x[...,x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim = -1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    '''
    将kv中多头注意力的头数复制n_rep份`   q
    '''
    batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    if n_rep == 1 : return x
    return(x[:, :, :, None, :].expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim).reshape
           (batch_size, seq_len, num_key_value_heads * n_rep, head_dim))

def precompute_freqs_cis(dim:int, end:int = int(32 * 1024), rope_base:float = 1e6, rope_scaling:dict = None):
    '''
    RoPE的预计算函数，集成了YaRN的长文本外推算法
    
    end: 推理看到的token长度
    rope_base: rope的基础数值，通常为100000    
    rope_scaling： 进行计算的缩放参数
    '''
    # 1、计算RoPE的基础角频率Θi = base ** -(2i/d)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # 2、YaRN长文本外推机制计算
    if rope_scaling is not None:
        '''
        orig_max: 原始训练的最大长度
        factor: 缩放因子
        beta_slow: 波长下限
        beta_fast: 波长上限
        '''
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("origina_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        # 如果目标长度 end 大于原始训练长度 orig_max，则触发 YaRN 缩放。factor 是缩放倍数
        if end / orig_max > 1.0:
            # inv_dim为维度索引的匿名函数，输入b参数，返回对应的维度索引。beta_fast/beta_slow返回低/高纬度索引，用于后续对不同索引段做不同的标记
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim  // 2 - 1) # 得到高低索引

            '''
            对于索引< low  区域设置为0 
            对于索引> high  区域设置为1
            中间区域平滑过渡 
            '''
            # torch.arange 生成(0,dim//2 - 1)，然后把低阈值的减去(- low)再进行归一化操作
            # torch.clamp((), 0, 1)将()里面数值限制到[0,1]之间，这里的限制是小于等于0的记作0， 大于等于1的记作1
            ramp = torch.clamp((torch.arange(dim // 2, device = freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 应用缩放，对于大于1的值（低频区域）缩小，实现插值
            freqs = freqs * (1 - ramp + ramp / factor) # 得到新的θ,至此YaRN完成。
    t = torch.arange(end, device = freqs.device) # token索引位置
    freqs = torch.outer(t, freqs).float() # t和freqs外积
    # 基于逐元素相乘构造旋转矩阵，计算旋转角度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor
    return freqs_cos, freqs_sin


class Attention(nn.Module):
    '''
    对语句向量进行注意力计算
    '''
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        '''
        GQA算法改进:Q有768为，而K和V仅有384维，
        生成时真正占显存的是缓存下来的 K 和 V，不是 Q。2 个 Q 头共用 1 组 K/V，KV cache 直接减半，
        质量损失通常很小。训练代码几乎感觉不到这个差别，代价主要在后面 repeat_kv 那一步补齐。
        '''
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # x: batch_size, sequence_length,
        bsz, seq_len, _ = x.shape

        # 1、对原来的x数据输入拆分成q k v三个向量矩阵
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2、切成多头注意力，多头注意力有助于模型学习多方面的关系，具体为
        '''
        xq: [B, S, 768] -> [B, S, 8, 96]
        xk/xv: [B, S, 384] -> [B, S, 4, 96]
        '''
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3、只对q k做归一化，这样q . k点积的时候不容爆掉
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 4、对q和k做位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 5、在推理阶段，kv存在的时候可以直接用缓存复用，提升计算速度
        if past_key_value is not None:
            # past_key_value里面第0维度是k，第1维度是v
            xk = torch.cat([past_key_value[0], xk],dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 6、将xq, xk, xv的1 2维度进行调换-> [B, Head_num, Seq_len, Head_dim]
        # 再将k v中的注意力头数量复制n_rep份，与xq的头数量对齐，保证顺利完成点积
        xq, xk, xv = (xq.transpose(1, 2),
                      repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 7、计算注意力
        has_padding = attention_mask is not None and not bool(torch.all(attention_mask == 1).item())
        use_flash = self.flash and seq_len > 1 and past_key_value is None and not has_padding
        if use_flash:
            # 封装好的attention计算
            output = F.scaled_dot_product_attention(xq, xk, xv,
                                                    dropout_p = self.dropout if self.training else 0.0,
                                                    is_causal = self.is_causal)
        else:
            scores = (xq @ xk.transpose(-1, -2) / math.sqrt(self.head_dim))
            # 不能偷看未来，将最后seq_len往后都设置为-inf，这样softmax之后均为0
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 对于句子的padding做掩码，对其句子长度统一的同时，将pad部分做attention后的0值处理
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim = -1)).type_as(xq) @ xv

        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output)) # 将多个多头注意力的分数进行投影输入最后结果
        return output, past_kv

class FeedForward(nn.Module):
    '''
    1、x:[B, S, hidden_size] -> [B, S, intermediate_size]
    2、对升维后的结果做做一次silu激活(self.acf_fn(self.gate_proj(x)))，再乘以激活前（升维后）的结果(self.up_proj(x))
    3、对成绩后的结果通过线性层降维，回到原来的[B, S, hidden_size]
    总的来说：将768向量通过线性层先升维，通过门控激活相乘后再降维。
    '''
    def __init__(self, config:MiniMindConfig, intermediate_size:int = None):
        super().__init__()
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.acf_fn = ACT2FN[config.hidden_act] # 配置中为silu激活函数

    def forward(self, x):
            return self.down_proj(self.acf_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    '''
     启动MOE情况下使用的前馈网络
    '''
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False) # 专家数为4
        # moe_intermediate_size与intermediate_size相同，均设置为768,这里设置了4个专家各自的前馈网络
        self.experts = nn.ModuleList([FeedForward(config,
                                                  intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act] # siLU

    def forward(self, x):

        # 1、对token向量展平
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim) # [B * S, H]

        # 2 、每个token专家打出的分数，并且进行归一化
        scores = F.softmax(self.gate(x_flat), dim = -1) # [B * S, num_experts]

        # 3、选择topk专家
        topk_weight, topk_idx = torch.topk(
            scores,
            k = self.config_num_experts_per_tok, # 每个token最多选择多少专家
            dim = -1,
            sorted=False
        ) # [B * S, config_num_experts_per_tok]， weight代表每个token的专家权重，idx表示每个token专家的id

        # 归一化处理，保证topk权重和为1
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim = -1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)

        # 4、对于专家选择的token输出结果
        for i, expert in enumerate(self.experts):
            # 查看哪些token 选择了当前专家
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim = -1).nozero().flatten() # 返回被当前i专家选中的token索引，如 [0, 1] 表示0和1token被第i个专家选中
                weight = topk_weight[mask].view(-1, 1) # 取出选择当前专家的权重，输出为[m, 1]，其中m为当前选择专家i的token数量，1里面储存的是权重值
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype)) # 累加专家权重的结果作为输出
            elif self.training:
                # 让expert参数假装参与计算，但是不做最终的改变
                y[0 ,0] += 0 * sum(p.sum() for p in expert.parameters())
        # 5、专家负载均衡损失，解决部分专家过分被使用，一些从不会被选择的问题
        if self.training and self.config.router_aux_loss_coef > 0:
            # 对每个token选择的专家结果做one_hot编码：[B * S, config_num_experts_per_tok] -> [B * S, config_num_experts_per_tok, num_experts]
            '''
            例子
            topk_idx =
                    [[1, 2],                [[0,1,0,0], [0,0,1,0]],
                     [0, 1],   ----->>      [[1,0,0,0], [0,1,0,0]],
                     [2, 3]]                [[0,0,1,0], [0,0,0,1]]
            '''
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)

            # 实际的均衡loss计算
            '''
            1) 得到的分数在所有token的维度先求平均，得到每个专家的路由平均概率 [B * S, num_experts] -> [1, num_experts]
            2) 乘以 load(每个专家实际被分配到的负载比例)，并且对每个专家进行求和sum()
            3)乘以专家数和设定好的均衡loss系数
            '''
            self.aux_loss = ((load * scores.mean(0)).sum() *
                             self.config.num_experts * self.config.router_aux_loss_coef)
        else:
            # 在非训练情况下不计算aux
            self.aux_loss = scores.new_zeros(1).squeeze() # 0
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    '''
    MiniMind的主题网络结构
    '''
    def __init__(self, layer_id:int, config:MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config) # 注意力层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 归一化层
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 归一化层化
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_kv_value = None,
                use_cache = False, attention_mask = None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings, past_kv_value, use_cache, attention_mask
        ) # 这里的hidden_states为注意力分数计算结果
        hidden_states += residual # 做残差链接
        # 再加上经过归一化+前馈网络的结果
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]) # Block有8层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask = None, past_kv_values = None,
                use_cache = False, **kwargs):
        batch_size, seq_length = input_ids.shape # [B, S]
        if hasattr(past_kv_values, 'layers'):
            past_kv_values = None
        past_kv_values = past_kv_values or [None] * len(self.layers)
        # 1、从kvcahce中获取当前token的绝对起始位置
        start_pos = past_kv_values[0][0].shape[1] if past_kv_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 2、获取rope的θ频率(包括yard步骤)
        # 如果rope计算的频率均为零，说明出现错误，需要重新计算
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim = self.config.head_dim,
                                                        end = self.config.max_position_embeddings,
                                                        rope_base = self.config.rope_theta,
                                                        rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 3、计算batch中所需要的位置部分
        position_embeddings = (self.freqs_cos[start_pos: start_pos + seq_length], self.freqs_sin[start_pos: start_pos + seq_length])

        presents = []
        # 4、应用到实际的MiniMindModules中，进行完整的transformers计算(MiniMindBlock类)
        for layer, past_kv_value in zip(self.layers, past_kv_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_kv_value = past_kv_value,
                use_cache = use_cache,
                attention_mask = attention_mask
            )
            presents.append(present) # 这里的present存储的是kv_cache
        # 进行归一化
        hidden_states = self.norm(hidden_states)

        # 5、计算MOE辅助损失
        '''
        1) 遍历所有layers，如果是MOEFeedForward层，计算auxloss，并且进行累加
        2） sum的第二个参数利用hidden_states新创建一个0标量张量，new_方法只继承原数据的类型和设备存放位置，不继承形状。目的是为了aux_loss计算结果也放在与hidden_states相同的设备和数据类型
        '''
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                       hidden_states.new_zeros(1).squeeze())

        # 返回transformers结果、kv cache，以及moe的aux_loss
        return hidden_states, presents, aux_loss



class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    '''
    因果语言模型，如何通过自回归的方式生成语言
    '''
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig # 配置文件
        super().__init__(self.config)
        self.model = MiniMindModel(self.config) # 定义模型网络
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) # 线形层
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask = None, past_kv_values = None, use_cache = False,
                logits_to_keep = 0, labels = None, **kwargs):
        '''
        训练/推理阶段的前向传播
        '''
        # 1、输入给transformers提取特征，输出最后的隐藏层表征、kv_cache以及moe_loss
        # hidden_states: [Batch_size, Seq_len, self.config.hidden_size]
        hidden_states, past_kv_values, aux_loss = self.model(input_ids, attention_mask, past_kv_values,
                                                             use_cache, **kwargs)

        # 2、确定对哪部分token进行logits计算
        # slice(-logits_to_keep, None) 等价于 [-logits_to_keep: ] 取最后logits_to_keep个token进行预测么可以减少自回归计算中显存的花销
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits是transformers模型对每个token针对整个词表原始未经归一化的打分分数，因此形状为[Batch_size, Seq_len, vocab_size]
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 3、计算交叉熵loss
        loss = None
        # labels指的是真实的答案序列 形状为 [batch_size, seq_len - 1]
        if labels is not None:
            # 用t时刻token的输出的logits 预测 t+1 时刻的label真实标签
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # x:[batch_size, seq_len - 1, vocab_size] -> 展平为 [batch_size * (seq_len - 1), vocab_size], x除去最后一个是因为最后一个token之后，句子已经结束了，不需要被预测
            # y:[batch_size, seq_len - 1] -> 展平为 [batch_size * (seq_len - 1)]。 y除去第一个是因为第一个token是直接提供的，不需要预测，因此也不需要答案
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        '''
        MoeCausalLMOutputWithPast返回了一个类，包括
        loss, aux_loss, logits （[batch_size, 1(最后预测的1一个token), vocab_size]）, past_key_values, hidden_states
        '''
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits = logits,
                                         past_key_values=past_kv_values, hidden_states=hidden_states)

    @torch.inference_mode
    def generate(self , inputs = None, attention_mask = None, max_new_tokens = 8192,
                 temperature = 0.85, top_p = 0.85, top_k = 50, eos_token_id = 2,
                 streamer = None, use_cache = True, num_return_sequence = 1, do_sample = True,
                 repetition_penalty = 1.0, **kwargs):

        # 1、初始化数据
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequence, 1) # 复制多条回答 [Batch_size, Seq_len]
        attention_mask = attention_mask.repeat(num_return_sequence, 1) if attention_mask is not None else None # 注意力掩码做同样的复制
        past_kv_values = kwargs.pop('past_kv_cache', None)
        finished = torch.zeros(input_ids.shape[0], dtype = torch.bool, device = input_ids.device) # 形状为[Batch_size]的全False张量作为训练的标记

        if streamer:
            streamer.put(input_ids.cpu())

        '''
        自回归主循环
        '''
        for _ in range(max_new_tokens):

            # 1、配合kvcache降低计算复杂度
            # past_kv_values[batchs_size, num_heads, past_seq_len, head_dim]
            past_len = past_kv_values[0][0].shape[1] if past_kv_values else 0 # 获取历史缓存过token的数量
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_kv_values, use_cache=use_cache, **kwargs) # 后续的轮数只对最后一个token（第past_len个）进行forward预测，用一个token根据kv缓存预测下一个token

            # 2、更新attention_mask
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None

            # 3、对结果做温度缩放
            logits = outputs.logits[:, -1, :] / temperature # 对最后一个结果（预测的token）做温度缩放 [Batch_size, vocab_size]

            # 4、 放置语言模型重复说话做惩罚
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): # input_ids[i]: [Seq_len]， 表示当前所有token的在表上的id
                    seen = torch.unique(input_ids[i]) # input_ids[i]表示当前批次的所有token，进行去重,得到一个去重后的一维张量
                    score = logits[i, seen] # 这里的score为这k个已经出现词对应的打分
                    '''
                    如果score > 0 , score / 惩罚因子
                    如果<0, score * 惩罚因子
                    目的是为了让已经出现过的 Token 在经过 Softmax 计算后，其被选中的概率降低。
                    对去重后已经出现过的token在词表中打分进行惩罚缩放，可以鼓励模型生成下一个词的时候，减少选择已经生成过词的概率
                    最后logits的形状没变[B, vocab_size]，但是已经出现过的词打分会经过放缩
                    '''
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # 5、topk截断,包括前k个最高分，其他设置为-inf，softmax后为0
            if top_k > 0:
                threshold = torch.topk(logits, top_k)[0][..., -1, None] # 对logits的vocab_size众多词表的概率中，选择前top_k个，再添加None进行广播，最后得到[batch_size, 1]的形状
                mask = logits < threshold
                logits[mask] = -float('inf') # 将小于top_k的分数都设置为-inf,方便后续softmax后为0
            # 6、top_p概率累计采样按概率从大到小排序，把词一个个加进来，直到累计概率刚好超过p就停止
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True) # 降序排序
                probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1) # 累加概率
                mask = cum_probs > top_p # 将累加大于top_p的部分全部做掩码设置为false
                mask[..., 1:] = mask[..., :-1].clone() # 右移动一位，保证最高概率的词会被永远保留
                mask[..., 0] = 0

                origin_mask = mask.scatter(1, sorted_indices, mask)
                logits[origin_mask] = -float('inf')

            # 7、对最终的打分进行softmax
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True) # [Batch_size, vocab_size]
            # 8、是否为结束符
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1),next_token.new_full((next_token.shape[0], 1),
                                                                                      eos_token_id), next_token)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            # 流式输出处理
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
            if streamer: streamer.end()
            if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
            return input_ids
