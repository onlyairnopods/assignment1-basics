"""
naive BPE 模块化实现
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


corpus = """low low low low low <|endoftext|>
lower lower widest widest widest <|endoftext|>
newest newest newest newest newest newest 
"""

# 1. initial vocabulary
def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    """
    vocab的key代表token (int ID), value代表token对应的 byte values
    """
    vocab: dict[int, bytes] = {x : bytes([x]) for x in range(256)}

    if special_tokens is not None:
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

    return vocab


def string_to_bytes(s: str, return_int: bool = False) -> tuple[bytes, ...] | tuple[int, ...]:
    """
    Example:
    >>> string_to_bytes("Hello")
    (b'H', b'e', b'l', b'l', b'o')
    >>> string_to_bytes("Hello", return_int=True)
    [72, 101, 108, 108, 111]
    """
    byte_array = s.encode("utf-8")
    if return_int:
        return tuple(list(byte_array))
    else:
        return tuple([bytes([x]) for x in byte_array])


# 2. pre-tokenize
def split_by_special_tokens(
    corpus: str, 
    special_tokens: list[str] | None = None,
) -> list[str]:
    """
    Example:
    >>> pre_tokenize("Hello World! <|endoftext|> bye", "<|endoftext|>")
    ["Hello World! ", "<|endoftext|>", " bye"]
    """
    # handle special tokens
    if special_tokens is not None:
        # NOTE 长→短排序，防止短的抢先匹配
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        # NOTE 使用 re.escape 转义 Special Token 中本来就有的 "|"，以免被误认为是正则表达式的“或”操作符
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        # print(pattern)
        # chunks = re.split(pattern, corpus)
        # NOTE pattern 和 f"(pattern)" 有区别, 在两边加上括号以在最后输出中保留 Special Token
        chunks = re.split(f"({pattern})", corpus)
    else:
        # 修复：即使没有 special tokens，也不能直接按空格 split，否则会丢失空格信息
        # 应该将整个 corpus 作为一个 chunk，交给下面的 PAT 处理
        chunks = [corpus] 
    
    return chunks


def pre_tokenize(
    corpus: str, 
    special_tokens: list[str] | None = None,
    drop_special_tokens: bool = True,
    encode_int: bool = True,
    return_dict: bool = True,
) -> dict[tuple[int | bytes, ...], int] | list[bytes | int]:
    """
    通过 pre-tokenization，我们把原始文本转换成许多“预分词片段”的 byte/id 序列，
    并用 Counter 统计每种片段出现的次数。
    后续在统计 pair 频率时，每个片段的相邻 token 对出现次数都会按其 count 加权累加，
    从而得到全语料的 pair 频次。
    """
    # handle special tokens
    chunks = split_by_special_tokens(corpus, special_tokens)
    # print(chunks)

    # pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if not return_dict:
        token_counts: list[tuple[bytes]] = []
        for chunk in chunks:
            if special_tokens and chunk in special_tokens and not drop_special_tokens:
                token_bytes = string_to_bytes(chunk, encode_int)
                token_counts.append(token_bytes)
            else:
                for match in re.finditer(PAT, chunk):
                    word = match.group(0)
                    token_bytes = string_to_bytes(word, encode_int)
                    token_counts.append(token_bytes)
        return token_counts

    else:
        word_counter: Counter[tuple[int | bytes, ...], int] = Counter()
        for chunk in chunks:
            # NOTE chunk 即 pre-token，不在整个语料库扫描统计，而是统计重复出现的 pre-token 的次数来加速。
            if special_tokens and chunk in special_tokens and not drop_special_tokens:
                token_bytes = string_to_bytes(chunk, encode_int)
                word_counter[token_bytes] += 1
            else:
                for match in re.finditer(PAT, chunk):
                    word = match.group(0)
                    token_bytes = string_to_bytes(word, encode_int)
                    word_counter[token_bytes] += 1
                    # NOTE 因为每个 word 在语料中出现了 word_counter[word] 次，所以 word 内部每出现一次 pair，就为全局频次贡献 word_counter[word]。
        return word_counter


# 3. count paris
def count_pairs(word_counter: dict[tuple[int | bytes, ...], int]) -> dict[tuple[int, int], int]:
    """
    统计文本中所有相邻字节对的出现频率。
    """
    # pair_counter: defaultdict[tuple[int, int], int] = defaultdict(int)
    pair_counter: Counter[tuple[int, int], int] = Counter()
    
    for tokens, count in word_counter.items():
        for tk1, tk2 in zip(tokens[:-1], tokens[1:]):
            pair_counter[(tk1, tk2)] += count

    return pair_counter


# 4.1 get most frequent pair
def get_most_frequent_pair(pair_counter: dict[tuple[int, int], int]) -> tuple[int, int]:
    """若多个 pair 频率相同，我们按 pair 的字典序（先比左 token，再比右 token）选择更大的那个。"""
    max_freq = max(pair_counter.values())
    candidates = []
    for pair, freq in pair_counter.items():
        if freq == max_freq:
            candidates.append(pair)
    best_pair = max(candidates)
    return best_pair


# 4.2 merge pairs, add to vocab
def add_pair_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> dict[int, bytes]:
    tk1, tk2 = pair
    assert tk1 in vocab and tk2 in vocab and \
        isinstance(tk1, int) and isinstance(tk2, int),\
              f"tk1={tk1}, tk2={tk2}"
    new_token_bytes: bytes = vocab[tk1] + vocab[tk2]
    vocab[len(vocab)] = new_token_bytes
    new_token = len(vocab) - 1
    return vocab, new_token


# 4.3 update corpus and recount pairs
def merge_pair_and_recount(
    word_counter: dict[tuple[int | bytes, ...], int],
    pair: tuple[int, int],
    new_token: int,
):
    new_word_counter: Counter[str, int] = Counter()
    new_pair_counter: Counter[str, int] = Counter()

    for tokens, count in word_counter.items():
        # tokens: a sequence of tokens of a chunk
        new_tokens = []
        i = 0
        L = len(tokens)

        while i < L:
            if i + 1 < L and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        new_word_counter[tuple(new_tokens)] += count

        # NOTE navie 方法，每次merge，都要算每个pair的count，效率低
        for tk1, tk2 in zip(new_tokens[:-1], new_tokens[1:]):
            new_pair_counter[(tk1, tk2)] += count

    return new_word_counter, new_pair_counter

    

def train_bpe(
    corpus: str,
    vocab_size: int = 263,
    special_tokens: list[str] | None = None,
):
    merges: defaultdict[tuple[int, int], int] = defaultdict(int) # 用于存储合并操作记录

    # Step 1: Initialize the vocabulary
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)

    # Step 2: Pre-tokenization
    # NOTE 记录每个 word/token 序列出现次数
    word_counter: Counter[tuple[int | bytes, ...], int] = pre_tokenize(corpus, special_tokens)

    # Step 3: Count pairs
    pair_counter: Counter[tuple[int, int], int] = count_pairs(word_counter)


    # Step 4: Iteratively merge pairs
    for _ in range(num_merges):
        # Step 4.1: Get the most frequent pair
        best_pair = get_most_frequent_pair(pair_counter)
        # print(f"({vocab[best_pair[0]]}, {vocab[best_pair[1]]})")

        # Step 4.2: Merge the pair and add to the vocabulary
        vocab, new_id = add_pair_to_vocab(vocab, best_pair)

        # Step 4.3: Update the corpus and recount pairs
        word_counter, pair_counter = merge_pair_and_recount(word_counter, best_pair, new_id)

        merges[best_pair] = new_id

        # print(f"Merge {vocab[best_pair[0]], vocab[best_pair[1]]} -> {new_id}")

    returned_merges: list[tuple[bytes, bytes]] = [(vocab[a], vocab[b]) for a, b in merges.keys()]

    return vocab, returned_merges




if __name__ == "__main__":
    vocab = init_vocab(['<|endoftext|>'])
    # print(vocab)

    # print(string_to_bytes('Hello', True))

    word_counter = pre_tokenize(corpus, ['<|endoftext|>'], encode_int=False)
    # print(word_counter)

    pair_counter = count_pairs(word_counter)
    # print(pair_counter)

    max_pair = get_most_frequent_pair(pair_counter)
    # print(max_pair)


    vocab, merges = train_bpe(
        corpus,
        vocab_size=263,
        special_tokens=['<|endoftext|>']
    )
    # print(vocab)
    print(len(vocab))
    print(merges)