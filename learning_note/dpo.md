# 数据类型

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
主要包含选择以及拒绝两类回答。

dpo的目的是为了模型的回答能够提高选择"chosen"答案的概率，降低选择"rejected"答案的概率

