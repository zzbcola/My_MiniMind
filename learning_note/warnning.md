# 加速模型训练的方法

## 方法一：sequence packing


痛点：

在传统的数据加载中，为了符合同一个batch里面长度一致才能实现矩阵计算的要求，通常需要设置一个‘max_seq_len’的值作为
模型训练时最长的序列长度，将数据集tokenizer之后不足max_seq_len的部分通过padding的方式补齐长度，
但是在进行attention的时候，虽然给padding的部分打上的attention mask，但是实际进行计算的时候，
这些mask的部分依然会被计算，导致算力的浪费。padding的长度越多，表示越多的计算被浪费。

解决方法：

从数据集入手，将数据长度拼接成接近模型训练中的‘max_seq_len’token值。

但这里有一点需要注意的是，通常需要document packing，对于如果加入的文本段token超出了max_seq_len，通常会进行缓存，
