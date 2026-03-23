"""
naive BPE 一口气实现
1. Initialization，将输入文本视为字节序列，每个字节作为一个token
2. Pretokenization，将文本中的特殊符号视为单独的token
3. Count Pairs: 统计文本中所有相邻字节对的出现频率。
4. Merge Pairs: 找到出现频率最高的相邻字节对，将它们合并为一个token。
    1. 获得频率最高的字节对
    2. 合并，将新的字节对加入词汇表
    3. 更新文本中所有出现该字节对的地方
    4. 重新统计文本中所有相邻字节对的频率
5. 重复步骤2、3，直到达到预定的词汇表大小或满足其他停止条件。
"""


import regex as re
from collections import defaultdict, Counter


def to_bytes(word: str) -> tuple[bytes, ...]:
    return tuple(
        [bytes([x]) for x in word.encode('utf-8')]
    )


def train_bpe(
    corpus: str,
    vocab_size: int = 263,
    special_tokens: list[str] = [],    
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. 初始化词表
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    special_token_bytes: list[bytes] = [token.encode('utf-8') for token in special_tokens]
    for token_bytes in special_token_bytes:
        if len(vocab) >= vocab_size:
            break  # 如果词表已满，停止添加特殊符号
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    
    # 2. Pre-tokenization
    # handle special tokens first
    if special_tokens:
        # NOTE 长→短排序，防止短的抢先匹配
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        # NOTE 使用 re.escape 转义 Special Token 中本来就有的 "|"，以免被误认为是正则表达式的“或”操作符
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        # NOTE pattern 和 f"(pattern)" 有区别, 在两边加上括号以在最后输出中保留 Special Token
        chunks = re.split(f"({pattern})", corpus)
    else:
        # 修复：即使没有 special tokens，也不能直接按空格 split，否则会丢失空格信息
        # 应该将整个 corpus 作为一个 chunk，交给下面的 PAT 处理
        chunks = [corpus] 
    # print(chunks)

    # pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_byte_sequence_counter: Counter[tuple[bytes, ...]] = Counter()
    for chunk in chunks:
        if chunk and chunk not in special_tokens: # 去除掉 special token
            for match in re.finditer(PAT, chunk):
                word: str = match.group(0)
                token_bytes_sequence: tuple[bytes, ...] = to_bytes(word) # 将字符串转为字节串
                # print(token_bytes_sequence)
                pre_tokenized_byte_sequence_counter[token_bytes_sequence] += 1
    # print(pre_tokenized_byte_sequence_counter)

    
    # 3. BPE 训练
    merges: list[tuple[bytes, bytes]] = [] # 用于记录合并操作，记录每次合并的两个 token bytes
    while len(vocab) < vocab_size:
        
        # 3.1 Count Pairs: 统计文本中所有相邻字节对的出现频率。
        pair_counter: Counter[tuple[bytes, bytes]] = Counter()
        for token_bytes_sequence, count in pre_tokenized_byte_sequence_counter.items():
            for i in range(len(token_bytes_sequence) - 1):
                pair_counter[(token_bytes_sequence[i], token_bytes_sequence[i + 1])] += count
        # print(pair_counter)

        if not pair_counter: # 如果没有可以合并的字节对了
            break

        # 3.2 Merge Pairs: 找到出现频率最高的相邻字节对
        # best_pair: tuple[bytes, bytes] = max(pair_counter, key=lambda x: pair_counter[x])
        # 找到频率最高且字典序最大的对进行合并
        max_freq = max(pair_counter.values())
        candidates = [
            pair for pair, freq in pair_counter.items() 
            if freq == max_freq
        ]
        best_pair: tuple[bytes, bytes] = max(candidates) # 字典序最大的对
        # print(best_pair)

        tk1, tk2 = best_pair[0], best_pair[1]

        # 3.3 合并得到新的字节token，将新的字节对加入词汇表
        new_token_bytes: bytes = tk1 + tk2
        # print(new_token_bytes)
        vocab[len(vocab)] = new_token_bytes # 将新的字节串添加到词汇表中

        # 3.4 更新 counter：原地替换所有包含 best_pair 的序列
        new_byte_sequence_counter: Counter[tuple[bytes, ...]] = Counter()
        for token_bytes_sequence, count in pre_tokenized_byte_sequence_counter.items():
            # 将序列中的 best_pair 替换为 new_token_bytes
            new_sequence = []
            i = 0
            while i < len(token_bytes_sequence):
                # 如果找到 best_pair，替换为新 token
                if (i + 1 < len(token_bytes_sequence) and 
                    token_bytes_sequence[i] == tk1 and token_bytes_sequence[i + 1] == tk2):
                    new_sequence.append(new_token_bytes)
                    i += 2
                else:
                    new_sequence.append(token_bytes_sequence[i])
                    i += 1
            
            new_byte_sequence_counter[tuple(new_sequence)] += count
        
        # 用新的 counter 替换旧的
        pre_tokenized_byte_sequence_counter = new_byte_sequence_counter
            
        # 3.5 记录合并操作
        merges.append((tk1, tk2))


    return vocab, merges

    


if __name__ == "__main__":
    corpus = """low low low low low <|endoftext|>
lower lower widest widest widest <|endoftext|>
newest newest newest newest newest newest 
"""

    # print(BYTES_TO_UNICODE_MAP)

    vocab, merges = train_bpe(corpus, vocab_size=263, special_tokens=["<|endoftext|>"])
    # print(vocab)
    print(len(vocab))
    print(merges)