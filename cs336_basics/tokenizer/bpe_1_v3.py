"""
naive gpt-2 style BPE
"""

import regex as re
from collections import defaultdict, Counter


def bytes_to_unicode() -> dict[int, str]:
    """
    将bytes(实际是0-255的整数)转换为Unicode字符, 调用函数直接就返回字典: {整数: unicode字符}
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    # print(bs)
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTES_TO_UNICODE: dict[int, str] = bytes_to_unicode() # 将整数转换为Unicode字符
UNICODE_TO_BYTES: dict[str, bytes] = {v: bytes([k]) for k, v in BYTES_TO_UNICODE.items()} # 将Unicode字符转换为字节


def pair_count(token_sequences: list[list[str]]) -> Counter[tuple[str, str]]:
    pair_counter: Counter[tuple[str, str]] = Counter()
    for sequence in token_sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counter[pair] += 1
    return pair_counter


def merge_pair_in_sequences(
    token_sequences: list[list[str]],
    pair_to_merge: tuple[str, str],
    new_token: str,
) -> list[list[str]]:
    """
    用 new_token 替换所有出现的 pair_to_merge。
    假设：
    token_sequences = [['h', 'e', 'l', 'l', 'o']]
    pair_to_merge = ('l', 'l')
    new_token = 'll'
    处理过程：
    遍历 ['h', 'e', 'l', 'l', 'o']
    前两个字节不是 ('l', 'l')，跳过
    到了下标2和3，发现是 ('l', 'l')，合并成 'll'
    结果变成 ['h', 'e', 'll', 'o']
    """
    new_sequences: list[list[str]] = []
    tk1, tk2 = pair_to_merge
    for sequence in token_sequences:
        new_sequence: list[str] = []
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and sequence[i] == tk1 and sequence[i + 1] == tk2:
                new_sequence.append(new_token)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        new_sequences.append(new_sequence)
    return new_sequences


def train_bpe(
    corpus: str,
    vocab_size: int = 263,
    special_tokens: list[str] = [],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Returns:
        vocab: The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
        to bytes (token bytes)

        merges: BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
        representing that <token1> was merged with <token2>.
        Merges are ordered by order of creation.
    """
    # 1. 初始化词表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)} # bytes() 是将整数列表转换为字节串的函数

    # 用一个集合来高效检查特殊符号的字节表示是否已存在于词汇表中
    existing_bytes = set(vocab.values())

    # 添加特殊符号到词表
    for special_token in special_tokens:
        if len(vocab) >= vocab_size:
            break  # 如果词表已满，停止添加特殊符号
        special_token_bytes = special_token.encode('utf-8') # 将特殊符号字符串转为字节串
        if special_token_bytes not in existing_bytes:
            vocab[len(vocab)] = special_token_bytes # 将新的字节串添加到词汇表中
            existing_bytes.add(special_token_bytes) # 记录

    # 2. Pre-tokenization
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

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 将文本转为unicode字符序列
    pre_tokenized_unicode_sequence: list[list[str]] = []
    for chunk in chunks:
        if chunk and chunk not in special_tokens: # 去除掉 special token
            for match in re.finditer(PAT, chunk):
                word: str = match.group(0)
                token_bytes: bytes = word.encode('utf-8') # 将字符串转为字节串
                # print(token_bytes)
                pre_tokenized_unicode_sequence.append(
                    [BYTES_TO_UNICODE[b] for b in token_bytes] # 将字节串转为unicode字符序列
                )
    # print(pre_tokenized_unicode_sequence)

    # 3. BPE 训练
    merges: list[tuple[bytes, bytes]] = [] # 用于记录合并操作，记录每次合并的两个 token bytes
    while len(vocab) < vocab_size:
        
        # 3.1 Count Pairs: 统计文本中所有相邻unicode字符对的出现频率。
        pair_counter = pair_count(pre_tokenized_unicode_sequence)
        if not pair_counter: # 如果没有可以合并的unicode字符对了
            break
        
        # 3.2 获得频率最高的unicode字符对
        # best_pair: tuple[str, str] = max(pair_counter, key=lambda x: pair_counter[x])
        # 找到频率最高且字典序最大的对进行合并
        max_freq = max(pair_counter.values())
        candidates = [
            pair for pair, freq in pair_counter.items() 
            if freq == max_freq
        ]
        best_pair: tuple[str, str] = max(candidates) # 字典序最大的对
        # print(best_pair)

        tk1, tk2 = best_pair[0], best_pair[1]

        # 3.3 合并得到新的unicode字符，将新的unicode字符对加入词汇表
        new_token_unicode: str = tk1 + tk2
        # print(f"Merging pair: ({tk1}, {tk2}) -> {new_token_unicode}")

        # 将新的unicode字符转换为字节串表示
        # new_token_bytes: bytes = new_token_unicode.encode('utf-8')
        # 将待合并的unicode字符转换为对应的字节表示
        tk1_bytes: bytes = UNICODE_TO_BYTES[tk1]
        tk2_bytes: bytes = UNICODE_TO_BYTES[tk2]

        # 将新的字节串添加到词汇表中
        new_token_bytes: bytes = tk1_bytes + tk2_bytes
        UNICODE_TO_BYTES[new_token_unicode] = new_token_bytes
        vocab[len(vocab)] = new_token_bytes

        merges.append((tk1_bytes, tk2_bytes)) # 记录合并操作

        # 3.4 更新文本中出现该新unicode字符序列的地方
        pre_tokenized_unicode_sequence = merge_pair_in_sequences(
            token_sequences=pre_tokenized_unicode_sequence,
            pair_to_merge=best_pair,
            new_token=new_token_unicode,
        )

    return vocab, merges

            





if __name__ == "__main__":
    corpus = """low low low low low <|endoftext|>
lower lower widest widest widest <|endoftext|>
newest newest newest newest newest newest 
"""

    # print(BYTES_TO_UNICODE_MAP)

    vocab, merges = train_bpe(corpus, special_tokens=["<|endoftext|>"])
    # print(vocab)
    print(merges)