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
    Policy model：要被训练的模型。一开始就是full_sft后的模型，但是允许参数更新。
    这里需要注意的是rollout中的policy模型和需要训练的policy模型，本质上都是同一个模型，但是为了能够达到“生成多个回答，根据回答里面reward模型打分较高的回答进行学习”这个grpo的核心目标，
    用rollout来生成多个回答，再用可学习的policy模型来生成这些token的概率，通过偏好奖励优质回答token概率。因此需要分成两个模型来看待。
    
    Reference model：从同一个 full_sft 权重加载，但冻结，用于 KL 约束。
    Reward model：默认使用外部奖励模型，路径如 internlm2-1_8b-reward

2、送入rollout引擎，让policy模型自己回答空缺的部分，并且生成多条(num_generate)，让外部模型对多条回答按照一定的规则打分给外部的reward模型进行打分(rewards)
    打分规则如下：
    回答长度：20 <= len <= 800，满足加分，否则扣分；
    如果有 </think>：- 思考内容长度合适，加分；
    - </think> 只出现一次，加分；
      - 否则扣分；
    n-gram 重复惩罚，降低重复输出。

3、计算所有回答打分的均值和标准差，然后针对每个回答计算优势advantage（这条回答比同组其他回答好多少？）
    advantage = (reward - group_mean) / group_std

4、对policy生成的完整文本（包括propmt和completion部分）重新送入ref模型（冻结参数的sft模型）和policy模型（没有冻结参数的sft模型）
    计算两个模型对于这些文本token生成的对数概率值ref_per_token_logps, per_token_logps

5、计算completion mask
    方便后续loss对非回答部分进行掩码

6、计算 KL 约束
    为了后训练的模型与sft后的原模型不能偏离太多

7、计算策略梯度，获取更新策略前后的回答质量差异
    loss的组成包括以下信息：
    ratio = torch.exp(per_token_logps - old_per_token_logps) 策略更新前后对于token的概率，衡量token选择概率的变化
    advantage 表示多个回答的优势
    ratio 表示当前策略相对旧策略对该回答的概率变化。
    clipped_radio 截断优势，防止优势过大导致激进策略
    KL散度 防止后训练的模型与sft后的原模型不能偏离太多


总的来说就是先输入上下文，让rollout引擎中的policy模型多次生成答案（让外部的reward模型对多个回答打分），
然后让ref模型的可学习的policy模型根据上下文，计算之前policy模型生成token的概率
计算新policy模型token概率和旧policy概率之差
