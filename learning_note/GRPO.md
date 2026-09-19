# 数据类型

    {
      "conversations": [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "回答一"},
        {"role": "user", "content": "问题二"},
        {"role": "assistant", "content": "回答二"},
        ....
        {"role": "assistant", "content": ""},
      ],
    }
rlaif.jsonl数据集用于GRPO阶段的对其偏好，数据集包含人类和聊天助手一问一答的多轮对话
数据集的特点在于最后一轮聊天助手进行回答的时候，内容为空，用于在GRPO后训练阶段依次执行以下步骤
1、初始化三个模型：
    Policy model：要被训练的模型。
    Reference model：从同一个 full_sft 权重加载，但冻结，用于 KL 约束。
    Reward model：默认使用外部奖励模型，路径如 internlm2-1_8b-reward

2、让模型自己回答空缺的部分，并且生成多条，让外部模型对多条回答按照一定的规则打分进行打分
    打分规则如下：
    回答长度：20 <= len <= 800，满足加分，否则扣分；
    如果有 </think>：- 思考内容长度合适，加分；
    - </think> 只出现一次，加分；
      - 否则扣分；
    n-gram 重复惩罚，降低重复输出。

3、计算所有回答打分的均值和标准差，然后针对每个回答计算优势advantage（这条回答比同组其他回答好多少？）
    advantage = (reward - group_mean) / group_std

4、计算 KL 约束
    为了后训练的模型与sft后的原模型不能偏离太多

5、计算策略梯度，获取更新策略前后的回答质量差异
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    old_per_token_logps：rollout 时旧策略生成回答的 log probability；
    per_token_logps：当前更新后策略重新计算同一回答的 log probability；
    ratio 表示当前策略相对旧策略对该回答的概率变化。
