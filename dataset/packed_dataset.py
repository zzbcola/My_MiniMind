'''
预训练数据加速：文档打包（document packing）+ 长度分桶分批（length-grouped batching）。

两种模式共用一份一次性生成的 token 缓存（默认放在数据同目录）：
    {stem}_tokens_u16.bin       所有文档 token 首尾相接的 token 流（uint16）
    {stem}_doc_lengths_i32.bin  每篇文档的 token 数（int32）
    {stem}_cache_meta.json      缓存元信息，防止用错 max_length

- packed  模式：按 max_length 把 token 流切成定长 pack，完全没有 padding
- grouped 模式：一条样本 = 一篇文档，把长度接近的文档放进同一个 batch，再做动态 padding
'''

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

CACHE_VERSION = 1


def _cache_paths(data_path, cache_dir=None):
    '''
    token缓存地址写入
    '''
    data_path = Path(data_path)
    cache_dir = Path(cache_dir) if cache_dir else data_path.parent
    stem = data_path.stem
    return (cache_dir / f'{stem}_tokens_u16.bin',
            cache_dir / f'{stem}_doc_lengths_i32.bin',
            cache_dir / f'{stem}_cache_meta.json')


def build_token_cache(data_path, tokenizer, max_length, cache_dir=None, rebuild=False,
                      tokenize_batch_size=4096, log_every=1000000):
    '''
    把预训练语料一次性 tokenize 成 token 流，并保存到磁盘，供后续训练时快速读取，避免每次训练都重新分词。
    预处理规则与原 PretrainDataset 保持一致：每篇 [bos] + tokens + [eos]，超出 max_length 截断。
    '''
    data_path = Path(data_path)
    tokens_path, lengths_path, meta_path = _cache_paths(data_path, cache_dir)
    meta = {'version': CACHE_VERSION, 'max_length': max_length, 'vocab_size': len(tokenizer)}

    if not rebuild and tokens_path.exists() and lengths_path.exists() and meta_path.exists():
        '''
        如果之前已经缓存过，就复用之前的缓存
        '''
        old = json.loads(meta_path.read_text())
        if old.get('version') == CACHE_VERSION and old.get('max_length') == max_length \
                and old.get('vocab_size') == len(tokenizer):
            print(f'[token cache] 复用已有缓存：{tokens_path}（{old.get("num_docs")} docs）')
            return tokens_path, lengths_path
        print('[token cache] 缓存元信息不匹配（max_length 或词表已变），重新构建')

    if len(tokenizer) >= 65536:
        raise ValueError('词表超过 uint16 上限，请把 dtype 改成 uint32')

    with open(data_path, 'r', encoding='utf-8') as fin:
        num_docs = sum(1 for _ in fin) # 获取数据长度
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    doc_lengths = np.zeros(num_docs, dtype=np.int32)
    meta['num_docs'] = num_docs

    tokens_tmp = Path(str(tokens_path) + '.tmp')
    lengths_tmp = Path(str(lengths_path) + '.tmp.npy')
    print(f'[token cache] 开始构建：{num_docs} 篇文档 -> {tokens_path}')

    cursor, written, pending = 0, 0, []

    def flush(fout):
        '''
        把当前累积的一批文本 批量 tokenize、加上特殊 token、记录长度、写入二进制 token 流。
        '''
        nonlocal cursor, written, pending # 已处理数量、 已写入token数量， 累计待处理数量
        if not pending:
            return
        batch = tokenizer(pending, add_special_tokens=False, truncation=True,
                          max_length=max_length - 2)['input_ids']
        buffer = []
        for ids in batch:
            ids = [bos_id] + ids + [eos_id]
            doc_lengths[cursor] = len(ids)
            cursor += 1
            buffer.extend(ids)
        np.asarray(buffer, dtype=np.uint16).tofile(fout)
        written += len(buffer)
        pending = []

    next_log = log_every
    with open(data_path, 'r', encoding='utf-8') as fin, open(tokens_tmp, 'wb') as fout:
        for line in fin:
            pending.append(json.loads(line)['text'])
            if len(pending) >= tokenize_batch_size:
                flush(fout)
                if cursor >= next_log:
                    print(f'[token cache] {cursor}/{num_docs} 篇, {written} tokens')
                    next_log += log_every
        flush(fout)

    np.save(lengths_tmp, doc_lengths)
    os.replace(tokens_tmp, tokens_path)
    os.replace(lengths_tmp, lengths_path)
    meta['num_tokens'] = written
    meta['mean_length'] = round(written / max(num_docs, 1), 3)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f'[token cache] 完成：{written} tokens / {num_docs} docs，平均 {meta["mean_length"]} tokens/doc')
    return tokens_path, lengths_path


class TokenCache:
    '''token 流 + 文档边界，packed / grouped 两种模式共用'''

    def __init__(self, data_path, tokenizer, max_length, cache_dir=None, rebuild=False):
        tokens_path, lengths_path = build_token_cache(
            data_path, tokenizer, max_length, cache_dir=cache_dir, rebuild=rebuild)
        self.tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r')
        self.doc_lengths = np.load(lengths_path)
        self.doc_offsets = np.zeros(self.doc_lengths.size + 1, dtype=np.int64)
        self.doc_offsets[1:] = np.cumsum(self.doc_lengths, dtype=np.int64)
        self.num_tokens = int(self.tokens.size)
        self.num_docs = int(self.doc_lengths.size)


class PackedPretrainDataset(Dataset):
    '''
    文档打包：把全部文档拼成一条 token 流，按 seq_len 切成定长 pack，完全没有 padding。
    每个 epoch 用 set_epoch(epoch) 重新打乱文档顺序，使每轮 pack 组合不同。
    '''

    def __init__(self, cache, seq_len, seed=42):
        self.cache = cache
        self.seq_len = seq_len
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        self.order = rng.permutation(self.cache.num_docs).astype(np.int64)
        self.cum = np.cumsum(self.cache.doc_lengths[self.order].astype(np.int64))
        self.num_packs = int(self.cum[-1]) // self.seq_len

    def __len__(self):
        return self.num_packs

    def __getitem__(self, index):
        start, stop = index * self.seq_len, (index + 1) * self.seq_len
        first = int(np.searchsorted(self.cum, start, side='right'))
        last = int(np.searchsorted(self.cum, stop - 1, side='right'))
        pieces, cursor = [], start
        for doc_index in range(first, last + 1):
            doc = int(self.order[doc_index])
            doc_start = int(self.cache.doc_offsets[doc])
            doc_end = int(self.cache.doc_offsets[doc + 1])
            base = int(self.cum[doc_index - 1]) if doc_index > 0 else 0
            offset = cursor - base
            take = min(doc_end - doc_start - offset, stop - cursor)
            pieces.append(np.asarray(
                self.cache.tokens[doc_start + offset: doc_start + offset + take], dtype=np.int64))
            cursor += take
            if cursor >= stop:
                break
        input_ids = torch.from_numpy(np.concatenate(pieces))
        return input_ids, input_ids.clone()


class PackedBatchSampler(Sampler):
    '''打包模式的 batch 采样器：每个 epoch 用 seed 打乱 pack 顺序，顺序可复现，续训安全'''

    def __init__(self, num_packs, batch_size, seed=42, shuffle=True):
        self.num_packs, self.batch_size, self.seed, self.shuffle = num_packs, batch_size, seed, shuffle

    def __len__(self):
        return math.ceil(self.num_packs / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            order = torch.randperm(self.num_packs, generator=generator).tolist()
        else:
            order = list(range(self.num_packs))
        for i in range(0, len(order), self.batch_size):
            yield order[i:i + self.batch_size]


class LengthGroupedPretrainDataset(Dataset):
    '''
    长度分桶模式：一条样本 = 一篇文档，配合 LengthGroupedBatchSampler + 动态 padding 使用。
    round_to=64 会把 batch 长度向上取整到 64 的倍数，产生少量 padding 但形状种类很少，便于 torch.compile。
    '''

    def __init__(self, cache, tokenizer, max_length, round_to=64):
        self.tokens = cache.tokens
        self.doc_offsets = cache.doc_offsets
        self.doc_lengths = cache.doc_lengths
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.round_to = round_to

    def __len__(self):
        return int(self.doc_offsets.size - 1)

    def __getitem__(self, index):
        start, stop = int(self.doc_offsets[index]), int(self.doc_offsets[index + 1])
        return self.tokens[start:stop].tolist()

    def collate_fn(self, batch):
        batch_max_length = min(max(len(tokens) for tokens in batch), self.max_length)
        if self.round_to and self.round_to > 1:
            batch_max_length = min(int(math.ceil(batch_max_length / self.round_to)) * self.round_to,
                                   self.max_length)
        input_ids_list, labels_list = [], []
        for tokens in batch:
            input_ids = tokens[:batch_max_length]
            input_ids = input_ids + [self.pad_token_id] * (batch_max_length - len(input_ids))
            labels = input_ids.copy()
            if len(tokens) < batch_max_length:
                labels[len(tokens):] = [-100] * (batch_max_length - len(tokens))
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        return torch.tensor(input_ids_list, dtype=torch.long), torch.tensor(labels_list, dtype=torch.long)


class LengthGroupedBatchSampler(Sampler):
    '''
    长度分桶 batch 采样器：每个 epoch 先全局打乱，再以 chunk_batches 个 batch 为一组，
    组内按长度排序后切 batch。排序只依赖 seed，因此 SkipBatchSampler 续训安全。
    '''

    def __init__(self, lengths, batch_size, seed=42, chunk_batches=64, shuffle=True):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.chunk_size = max(batch_size, batch_size * chunk_batches)
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            order = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            order = list(range(len(self.lengths)))
        chunks = [order[i:i + self.chunk_size] for i in range(0, len(order), self.chunk_size)]
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + 1)
            chunk_order = torch.randperm(len(chunks), generator=generator).tolist()
        else:
            chunk_order = list(range(len(chunks)))
        for chunk_index in chunk_order:
            chunk = sorted(chunks[chunk_index], key=self.lengths.__getitem__)
            for i in range(0, len(chunk), self.batch_size):
                yield chunk[i:i + self.batch_size]