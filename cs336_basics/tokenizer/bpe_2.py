"""
naive BPE 模块化实现
1. Initialization，将输入文本视为字节序列，每个字节作为一个token
2. # NOTE Pretokenization，将文本中的特殊符号视为单独的token
    -->
    1. 改成多进程
    2. 从路径读取文件
3. # NOTE Count Pairs: 统计文本中所有相邻字节对的出现频率。
    -->
    1. cache bytes pair count，通过索引字节对的计数，改成增量更新
4. Merge Pairs: 找到出现频率最高的相邻字节对，将它们合并为一个token。
    1. 获得频率最高的字节对
    2. 合并，将新的字节对加入词汇表
    3. 更新文本中所有出现该字节对的地方
    4. 重新统计文本中所有相邻字节对的频率
5. 重复步骤2、3，直到达到预定的词汇表大小或满足其他停止条件。
"""

import os
import regex as re
from collections import defaultdict, Counter
from tqdm import tqdm

from multiprocessing import Process, Queue
from collections import Counter

from cs336_basics.tokenizer.utils import find_chunk_boundaries, timetracker

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


@timetracker
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
            if special_tokens and chunk in special_tokens:
                if not drop_special_tokens:
                    token_bytes = string_to_bytes(chunk, encode_int)
                    token_counts.append(token_bytes)
                # else: skip special tokens entirely when drop_special_tokens=True
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
            if special_tokens and chunk in special_tokens:
                if not drop_special_tokens:
                    token_bytes = string_to_bytes(chunk, encode_int)
                    word_counter[token_bytes] += 1
                # else: skip special tokens entirely when drop_special_tokens=True
            else:
                for match in re.finditer(PAT, chunk):
                    word = match.group(0)
                    token_bytes = string_to_bytes(word, encode_int)
                    word_counter[token_bytes] += 1
                    # NOTE 因为每个 word 在语料中出现了 word_counter[word] 次，所以 word 内部每出现一次 pair，就为全局频次贡献 word_counter[word]。
        return word_counter


# NOTE new added
def pre_tokenize_worker(*args):
    try:
        input_path, special_tokens, queue, start, end, drop_special_tokens, encode_int, return_dict = args

        # Read the chunk from the file
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_size = end - start
            chunk_bytes = f.read(chunk_size)
            chunk: str = chunk_bytes.decode("utf-8", errors="ignore")

        word_counter = pre_tokenize(
            chunk,
            special_tokens,
            drop_special_tokens,
            encode_int,
            return_dict,
        )

        # Put the result in the queue
        queue.put(word_counter)
    except Exception as e:
        import traceback
        print(f"Worker error: {e}")
        print(traceback.format_exc())
        # Still put something in the queue to avoid blocking
        queue.put(Counter())


# 3. count paris
# def count_pairs(word_counter: dict[tuple[int | bytes, ...], int]) -> dict[tuple[int, int], int]:
#     """
#     统计文本中所有相邻字节对的出现频率。
#     """
#     # pair_counter: defaultdict[tuple[int, int], int] = defaultdict(int)
#     pair_counter: Counter[tuple[int, int], int] = Counter()
    
#     for tokens, count in word_counter.items():
#         for tk1, tk2 in zip(tokens[:-1], tokens[1:]):
#             pair_counter[(tk1, tk2)] += count

#     return pair_counter

# NOTE updated
def count_pairs(
    word_counter: dict[tuple[int | bytes, ...], int],
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> dict[tuple[int, int], int]:
    
    pair_counter: Counter[tuple[int, int], int] = Counter()

    for word, count in word_counter.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])

            # NOTE 记录该 pair 出现在哪些 word（token 序列）里, 
            # 这个映射非常关键：
            # 当我们选择某个 pair 进行 merge 时，
            # 只有包含该 pair 的 word 会发生变化。
            # 借助 pair_to_words，
            # 我们可以只遍历这些“受影响的 words”，
            # 并对 pair_counter 做**局部增量更新**，
            # 而不是每轮都重新扫描全部 word_counter。
            # 这样当我们决定 merge 某个 pair 时，就只需要遍历 pair_to_words[pair] 里的那一小部分 word，而不必全量扫描所有 word。
            pair_to_words[pair].add(word)
            # 我们还需要在 merge 之后，更新这个索引：
            # 当某个 pair 被 merge 成一个新 token 后，
            # 所有包含该 pair 的 word 都会发生变化，
            # 因此我们需要把这些 word 从旧 pair 的索引里移除，
            # 并把它们添加到新 pair 的索引里。

            # NOTE 记录该相邻 pair 在全语料中的总出现次数。 
            # 因为每个 word 在语料中出现了 word_counter[word]=count 次，
            # 所以 word 内部每出现一次 pair，就为全局频次贡献 word_counter[word]=count。
            pair_counter[pair] += count
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


# NOTE new added 使用最大堆维护当前频率最高的pair
import heapq
class HeapItem:
    def __init__(self, neg_freq: int, pair_bytes: tuple[bytes, bytes], pair: tuple[int, int]):
        self.neg_freq = neg_freq
        self.pair_bytes = pair_bytes
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair_bytes > other.pair_bytes  # reverse order for max-heap behavior


def build_pair_heap(pairs_counter: Counter[tuple[int, int], int], vocab: dict[int, bytes]) -> list[HeapItem]:
    heap = []
    for (tk1, tk2), f in pairs_counter.items():
        if f > 0:
            item = HeapItem(-f, (vocab[tk1], vocab[tk2]), (tk1, tk2)) # 用负号把它变成“最大堆”，例如存成：key = (-freq, -a, -b)
            heapq.heappush(heap, item)
    return heap


def pop_most_frequent_pair(heap, pair_counter: Counter[tuple[int, int], int]) -> tuple[int, int]:
    while heap:
        item = heap[0]  # Peek at the top item
        neg_f = item.neg_freq
        pair = item.pair
        cur_f = pair_counter.get(pair, 0)
        # NOTE 频次在 merge 之后会发生变化，因此堆里旧的条目可能变“过期”。
        # 在 pop 堆顶时，我们需要检查该 pair 的当前频次是否和堆里存的频次一致；
        # 如果不一致，说明堆顶是过期的，就继续 pop 直到找到一个有效的 pair。
        if cur_f <= 0 or -neg_f != cur_f:  # frequency changed, which means the pair we store in heap is stale
            heapq.heappop(heap)
            continue
        return pair

    raise ValueError("No positive-frequency pairs remain")


# 4.2 merge pairs, add to vocab
def add_pair_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> dict[int, bytes]:
    tk1, tk2 = pair
    assert tk1 in vocab and tk2 in vocab and isinstance(tk1, int) and isinstance(tk2, int), f"tk1={tk1}, tk2={tk2}"
    new_token_bytes: bytes = vocab[tk1] + vocab[tk2]
    vocab[len(vocab)] = new_token_bytes
    new_token = len(vocab) - 1
    return vocab, new_token


# 4.3 update corpus and recount pairs
# def merge_pair_and_recount(
#     word_counter: dict[tuple[int | bytes, ...], int],
#     pair: tuple[int, int],
#     new_token: int,
# ):
#     new_word_counter: Counter[str, int] = Counter()
#     new_pair_counter: Counter[str, int] = Counter()

#     for tokens, count in word_counter.items():
#         # tokens: a sequence of tokens of a chunk
#         new_tokens = []
#         i = 0
#         L = len(tokens)

#         while i < L:
#             if i + 1 < L and (tokens[i], tokens[i + 1]) == pair:
#                 new_tokens.append(new_token)
#                 i += 2
#             else:
#                 new_tokens.append(tokens[i])
#                 i += 1
        
#         new_word_counter[tuple(new_tokens)] += count

#         # NOTE navie 方法，每次merge，都要算每个pair的count，效率低
#         for tk1, tk2 in zip(new_tokens[:-1], new_tokens[1:]):
#             new_pair_counter[(tk1, tk2)] += count

#     return new_word_counter, new_pair_counter


# NOTE
def get_new_word(
    word: tuple[int | bytes, ...],
    target_pair: tuple[int, int],
    new_token: int,
) -> tuple[int | bytes, ...]:
    new_word = []
    i = 0
    L = len(word)

    while i < L:
        if i + 1 < L and (word[i], word[i + 1]) == target_pair:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


# NOTE 
def merge_pair_with_heap_index(
    word_counter: dict[tuple[int | bytes, ...], int],
    pair_counter: Counter[tuple[int, int], int],
    target_pair: tuple[int, int],
    new_token: int,
    vocab: dict[int, bytes],
    pair_heap: list,
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> tuple[
    dict[tuple[int | bytes, ...], int],
    Counter[tuple[int, int], int],
    list,
    dict[tuple[int, int], set[tuple[int, ...]]],
]:
    """
    在 merge 之后，更新这个索引：
    当某个 pair 被 merge 成一个新 token 后，
    所有包含该 pair 的 word 都会发生变化，
    因此我们需要把这些 word 从旧 pair 的索引里移除，
    并把它们添加到新 pair 的索引里。
    """
    # Start from full counters so unaffected words remain.
    new_word_counter = word_counter.copy()
    updated_pair_counter = pair_counter.copy()
    changed_pairs: set[tuple[int, int]] = set()

    # Get all words that contain the target pair.
    affected_words = list(pair_to_words.get(target_pair, set()))

    for word in affected_words:
        freq = word_counter.get(word, 0)
        if freq <= 0 or len(word) < 2:
            continue

        # 1. Remove the old word from the corpus counts.
        new_word_counter[word] -= freq
        if new_word_counter[word] <= 0:
            del new_word_counter[word]

        # 2. Subtract ALL old adjacent pairs for this word + remove old word from index.
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            updated_pair_counter[pair] -= freq
            changed_pairs.add(pair)

            s = pair_to_words.get(pair)
            if s is not None:
                s.discard(word)
                if not s:
                    del pair_to_words[pair]

        # 3. Build merged word (greedy left-to-right, same as standard BPE).
        new_word = get_new_word(word, target_pair, new_token)
        new_word_counter[new_word] += freq

        # 4. Add ALL new adjacent pairs for merged word + add merged word into index.
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                updated_pair_counter[pair] += freq
                changed_pairs.add(pair)
                pair_to_words.setdefault(pair, set()).add(new_word)

    # 5. Push updated frequencies for changed pairs into heap (skip non-positive).
    if pair_heap is not None:
        for pair in changed_pairs:
            freq = updated_pair_counter.get(pair, 0)
            if freq > 0:
                heapq.heappush(pair_heap, HeapItem(-freq, (vocab[pair[0]], vocab[pair[1]]), pair))

    return new_word_counter, updated_pair_counter, pair_heap, pair_to_words

    
# NOTE updated
@timetracker
def train_bpe(
    input_path: str,
    vocab_size: int = 263,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
    **kwargs,
):

    merges: defaultdict[tuple[int, int], int] = defaultdict(int) # 用于存储合并操作记录

    # Step 1: Initialize the vocabulary
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)


    # Step 2: Pre-tokenization
    # 2.1 Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=kwargs.get("desired_num_chunks", 2),
            split_special_token=b"<|endoftext|>"
        )

    if verbose:
        print(f"train_bpe called with input_path={input_path}, vocab_size={vocab_size}", flush=True)
        print(f"Initialized vocab with {len(vocab)} tokens, need {num_merges} merges", flush=True)
        print(f"Found {len(chunk_boundaries)-1} chunks with boundaries: {chunk_boundaries[:5]}...")

    # 2.2 Count word frequencies across chunks using multiprocessing
    queue = Queue()
    processes: list[Process] = []

    for i, (start, end) in enumerate(zip(chunk_boundaries[:-1], chunk_boundaries[1:])):
        if verbose:
            print(f"Starting worker {i+1} for bytes {start} to {end} ({(end-start)/(1024*1024):.2f} MB)")
        p = Process(
            target=pre_tokenize_worker,
            args=(
                input_path,
                special_tokens,
                queue,
                start,
                end,
                True, # drop_special_tokens
                True, # encode_int
                True, # return_dict
            ),
        )
        processes.append(p)
        p.start()

    # word_counter = pre_tokenize(corpus, special_tokens)
    # NOTE 记录每个 word/token 序列出现次数
    word_counter: Counter[tuple[int | bytes, ...], int] = Counter()
    for i in tqdm(range(len(processes)), desc="Collecting worker results", disable=not verbose):
        try:
            # Large files need more time - allow 20 minutes per worker
            partial_counter = queue.get(timeout=600 * 2)
            word_counter.update(partial_counter)
            # print(f"Worker {i+1}/{len(processes)} completed successfully", flush=True)
        except Exception as e:
            print(f"Warning: Worker {i+1} failed with error: {e}", flush=True)
            continue

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Warning: Process exited with code {p.exitcode}")

    # print(f"{word_counter=}")

    # Step 3: Count pairs
    # pair_counter: Counter[tuple[int, int], int] = count_pairs(word_counter)
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    pair_counter: Counter[tuple[int, int], int] = count_pairs(word_counter, pair_to_words)

    # Step 4: BPE core loop Iteratively merge pairs
    pair_heap: list[HeapItem] = build_pair_heap(pair_counter, vocab)
    
    for _ in tqdm(range(num_merges), desc="Merging pairs", disable=not verbose):
        # Step 4.1: Get the most frequent pair
        # best_pair = get_most_frequent_pair(pair_counter)
        best_pair = pop_most_frequent_pair(pair_heap, pair_counter)
        # print(f"({vocab[best_pair[0]]}, {vocab[best_pair[1]]})")

        # Step 4.2: Merge the pair and add to the vocabulary
        vocab, new_id = add_pair_to_vocab(vocab, best_pair)

        # Step 4.3: Update the corpus and recount pairs
        # word_counter, pair_counter = merge_pair_and_recount(word_counter, best_pair, new_id)
        word_counter, pair_counter, pair_heap, pair_to_words = merge_pair_with_heap_index(
            word_counter,
            pair_counter,
            best_pair,
            new_id,
            vocab,
            pair_heap,
            pair_to_words,
        )

        merges[best_pair] = new_id

        # print(f"Merge {vocab[best_pair[0]], vocab[best_pair[1]]} -> {new_id}")

    returned_merges: list[tuple[bytes, bytes]] = [(vocab[a], vocab[b]) for a, b in merges.keys()]


    if kwargs.get("save_path"):
        save_vocab_and_merges(vocab, returned_merges, kwargs["save_path"])
        with open(os.path.join(kwargs["save_path"], "special_tokens.txt"), "w", encoding="utf-8") as f:
            if special_tokens:
                for token in special_tokens:
                    f.write(f"{token}\n")

    return vocab, returned_merges


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
):
    import json
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_filepath = os.path.join(output_dir, "vocab.json")
    merges_filepath = os.path.join(output_dir, "merges.txt")

    # Save vocab
    vocab_inv = {v.decode("latin1"): k for k, v in vocab.items()}
    with open(vocab_filepath, "w") as vf:
        json.dump(vocab_inv, vf, ensure_ascii=False, indent=2)

    # Save merges
    with open(merges_filepath, "w") as mf:
        # mf.write("#version: 0.2\n")
        for a, b in merges:
            mf.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")



if __name__ == "__main__":
    vocab = init_vocab(['<|endoftext|>'])
    # print(vocab)

    # print(string_to_bytes('Hello', True))

    word_counter = pre_tokenize(corpus, ['<|endoftext|>'], encode_int=False)
    # print(word_counter)

    # pair_counter = count_pairs(word_counter)
    # print(pair_counter)

    # max_pair = get_most_frequent_pair(pair_counter)
    # print(max_pair)

    # vocab, merges = train_bpe(
    #     corpus,
    #     vocab_size=263,
    #     special_tokens=['<|endoftext|>']
    # )
    vocab, merges = train_bpe(
        input_path="./test_corpus.txt",
        vocab_size=263,
        special_tokens=['<|endoftext|>'],
        desired_num_chunks=2,
        # save_path="./test_bpe_output",
    )
    # print(vocab)
    print(len(vocab))
    print(merges)