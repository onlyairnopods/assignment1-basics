from typing import Iterable, Iterator
import json
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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



class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, 
        list of merges, 
        and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # support user provided special tokens
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [st.encode("utf-8") for st in self.special_tokens]
        
        for token in self.special_tokens_bytes:
            if token not in self.vocab_inv:
                new_id = len(self.vocab)
                self.vocab[new_id] = token
                self.vocab_inv[token] = new_id

        self.merges_rank: dict[tuple[bytes, bytes], int] = dict(zip(merges, range(len(merges))))

        self.eos_token_id = self.vocab_inv.get(b"<|endoftext|>", None)


        # encode_fast()函数需要用到
        # 预计算 encode_fast 需要的字典，避免每次调用都重新构建
        self.rank: dict[tuple[int, int], int] = {}
        self.merge_to_new_id: dict[tuple[int, int], int] = {}

        for r, (a_bytes, b_bytes) in enumerate(self.merges):
            a_id = self.vocab_inv.get(a_bytes)
            b_id = self.vocab_inv.get(b_bytes)
            new_id = self.vocab_inv.get(a_bytes + b_bytes)
            if a_id is None or b_id is None or new_id is None:
                continue
            pair = (a_id, b_id)
            self.rank[pair] = r
            self.merge_to_new_id[pair] = new_id


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens (user provided special tokens).
        """
        # load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            vocab: dict[int, bytes] = {int(i): bytes(v, "latin1") for v, i in data.items()}

        # load merges
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((bytes(parts[0], "latin1"), bytes(parts[1], "latin1")))

        # support user provided special tokens

        return cls(vocab, merges, special_tokens)
    

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""

        token_ids: list[int] = []

        # 1. 最优先处理特殊符号
        # Sort special tokens by length (longest first) to avoid partial matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]
        # chunks = split_by_special_tokens(text, self.special_tokens)

        for chunk in chunks:
            if chunk == "": # 空文本块跳过
                continue
            if chunk in self.special_tokens:
                # if it's a special token, add its ID directly 如果是特殊token，直接encode
                token_ids.append(self.vocab_inv[chunk.encode("utf-8")])
            else:
                # 处理普通文本用BPE编码
                # 2. pre-tokenize 将其预分割为多个 words
                words = [m.group(0) for m in re.finditer(PAT, chunk)]
                for word in words:
                    # convert token to bytes tuple
                    pre_token = string_to_bytes(word) # a sequence of UTF-8 bytes

                    # 3. apply BPE merges
                    merged_bytes = self._apply_merges(pre_token)
                    # print(merged_bytes)

                    # concat token ids
                    token_ids.extend(self.vocab_inv[b] for b in merged_bytes)

        return token_ids

    
    def _apply_merges(self, pre_token: tuple[bytes, ...]) -> list[bytes]:
        """
        Apply BPE merges to a sequence of bytes.

        Args:
            byte_tuple: A tuple of single-byte tokens.

        Returns:
            A list of merged byte tokens after applying all applicable merges.
        """
        # a sequence of UTF-8 bytes
        word: list[bytes] = list(pre_token)

        # 获取所有相邻的对
        byte_pairs = lambda w: set(
            (w[i], w[i+1]) for i in range(len(w) - 1)
        )
        # 等价于：
        # def get_pairs(word: list[bytes]) -> set[tuple[bytes, bytes]]:
        #     pairs = set()
        #     for i in range(len(word) - 1):
        #         pairs.add((word[i], word[i + 1]))
        #     return pairs


        # 合并
        while True:
            candidate_pairs: set[tuple[bytes, bytes]] = byte_pairs(word)
            ranked_pairs: list[tuple[int, tuple[bytes, bytes]]] = []
            for pair in candidate_pairs:
                if pair in self.merges_rank: # 迭代遍历 merges 列表查找最靠前的合并
                    # 检查合并后的 token 是否在词汇表中
                    merged_token = pair[0] + pair[1]
                    if merged_token in self.vocab_inv:
                        ranked_pairs.append(
                            (self.merges_rank[pair], pair)
                        )
            if not ranked_pairs:
                break

            # 查找rank最小，即迭代地、贪婪地合并最先出现的换
            _, best_pair = min(ranked_pairs)


            # 等价于调用 训练时的 get_new_word()
            new_word: list[bytes] = []
            i = 0
            L = len(word)

            while i < L:
                if i + 1 < L and (word[i], word[i + 1]) == best_pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

        return word

    def encode_fast(self, text: str) -> list[int]:
        """ (还没细看)
        通过 min Heap 和 Double Linked List 来高效实现这个Encode

        用一个min heap，来获取我们最先要实现merge的pair，也就是在训练阶段，出现频率最高的pair。
        """

        def _pre_tokenize(text: str) -> list[bytes]:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(map(re.escape, sorted_special_tokens))
            if pattern:
                chunks = re.split(f"({pattern})", text)
            else:
                chunks = [text]

            token_list: list[bytes] = []

            for chunk in chunks:
                if chunk == "":
                    continue
                if chunk in self.special_tokens:
                    token_list.append(chunk.encode("utf-8"))
                else:
                    for tok in [m.group(0) for m in re.finditer(PAT, chunk)]:
                        # Each regex token becomes a single bytestring.
                        token_list.append(tok.encode("utf-8"))

            return token_list

        import heapq
        def merge_one_pretoken(ids: list[int]) -> list[int]:
            n = len(ids)
            if n <= 1:
                return ids

            alive = [True] * n

            # Doubly-linked list over positions 0..n-1 (positions are stable; nodes get "deleted")
            prev = [-1] * n
            nxt = [-1] * n
            for i in range(n):
                prev[i] = i - 1
                nxt[i] = i + 1 if i + 1 < n else -1

            # best pair per left-position i: (rank, i)
            heap: list[tuple[int, int]] = []

            def push_if_valid(i: int):
                cur_r = None
                j = nxt[i]
                if j == -1 or not alive[i] or not alive[j]:
                    cur_r = None
                else:
                    cur_r = self.rank.get((ids[i], ids[j]))

                if cur_r is not None:
                    heapq.heappush(heap, (cur_r, i))

            for i in range(n):
                push_if_valid(i)

            while heap: # 只要还有候选 pair，就继续尝试合并
                r, i = heapq.heappop(heap) # 取出当前 rank 最小的候选：(rank, 左端点位置 i)
                j = nxt[i] # 右端点位置 j 是 i 在链表中的后继
                if j == -1 or not alive[i] or not alive[j]: # i/j 无效或 i 已到尾部：这是过期候选
                    continue # 跳过，继续处理下一个堆元素
                
                # stale check: rank might no longer match current neighbor 
                # 堆里的记录可能已过期（邻居关系/ids 已改变），需要重新验证
                pair = (ids[i], ids[j]) # 当前时刻 i 和 j 对应的 token id 组成的相邻 pair
                cur_r = self.rank.get(pair) # 查询这个 pair 在 merge 规则中的 rank（不可合并则为 None）
                if cur_r is None or cur_r != r: # 现在不可合并，或 rank 已不匹配：说明堆元素过期
                    continue # 跳过该候选

                # merge i and j into i (use precomputed mapping to avoid KeyError)
                # 执行合并：把 (ids[i], ids[j]) 合成一个新 token，并写回到位置 i
                new_id = self.merge_to_new_id.get(pair) # 查找该 pair 合并后的 token id
                if new_id is None: # 理论上不该发生（rank 有但映射没建好），当作过期/异常处理
                    continue # 跳过

                ids[i] = new_id # 用新 token id 覆盖左端点 i（i 成为合并后的节点）

                # delete j from the linked list
                # 从链表中删除 j：j 被 i 吞掉了
                alive[j] = False # 标记 j 节点被删除
                nj = nxt[j] # 记住 j 的后继节点
                nxt[i] = nj # 让 i 直接指向 nj（跳过 j）
                if nj != -1: # 如果 nj 存在
                    prev[nj] = i # 更新 nj 的前驱为 i，保持链表一致

                # Only pairs that can change are around i (prev[i], i) and (i, nxt[i])
                # 局部更新：合并只会影响 i 附近的两个相邻 pair
                pi = prev[i] # i 的前驱位置
                if pi != -1: # 如果前驱存在
                    push_if_valid(pi) # (pi, i) 这个 pair 可能变得可合并或 rank 改变
                push_if_valid(i) # (i, nxt[i]) 这个 pair 也可能变得可合并或 rank 改变

            # materialize result by walking the linked list
            out: list[int] = []          # 最终合并后的 token id 序列
            k = 0                        # 从链表头（位置 0）开始遍历
            while k != -1:               # -1 表示到达链表末尾
                if alive[k]:             # 如果该位置还没有被合并删除
                    out.append(ids[k])   # 把当前位置的 token id 加入输出
                k = nxt[k]               # 跳到下一个“仍在链表中的”位置
            return out

        # 1. Pre-tokenization：先粗粒度切分文本
        byte_tokens = _pre_tokenize(text)

        # 2. 对每个 pre-token 做 BPE merge
        token_ids: list[int] = []
        for btok in byte_tokens:
            if btok in self.special_tokens_bytes:
                token_ids.append(self.vocab_inv[btok])
            else:
                ids = [self.vocab_inv[bytes([b])] for b in btok]
                token_ids.extend(merge_one_pretoken(ids))

        return token_ids
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定一个字符串可迭代对象，返回一个惰性产生token ID的生成器。
        适用于内存高效地处理大文件。

        Given an iterable of strings (e.g., a file handle), yield token IDs lazily.
        
        Args:
            iterable: An iterable source of text chunks.
            
        Yields:
            Token IDs generated by processing the input iterable.
        """
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        # ids to bytes and then concatenate
        full_bytes = b"".join(self.vocab.get(token_id, "\ufffd".encode("utf-8")) for token_id in ids)

        # bytes to string, replacing invalid utf-8 sequences
        text = full_bytes.decode("utf-8", errors="replace")
        return text
    

import numpy as np
import os
from tqdm import tqdm
def encode_file_to_bin(
    tokenizer: BPETokenizer,
    text_path: str,
    output_bin_path: str,
    dtype=np.uint16,
):
    """
    通常会希望把一整个文本文件编码成紧凑的二进制（.bin），
    方便后续训练时用 np.memmap 之类的方式高效加载，而不是每次都重新分词。
    
    这段函数做的事情很简单：按行读取文本 → 把每行编码成 token ids → 用固定 dtype 写入二进制文件。
    
    为什么用 unit16 就可以了？
    在BPE的训练阶段，我们将vocab size设置为 10,000 或者 32,000 远远小于 uint16的最大值 65,535因此用uint16是安全的。
    """
    total_bytes = os.path.getsize(text_path)

    with open(text_path, "r", encoding="utf-8") as f_in, open(output_bin_path, "wb") as f_out:
        p_bar = tqdm(
            total=total_bytes,
            desc=f"Encoding {os.path.basename(text_path)} to binary: {os.path.basename(output_bin_path)}",
            unit_scale=True,
            unit="B",
        )
        for line in f_in:
            token_ids = tokenizer.encode_fast(line)          # 1) 把一行文本编码成 token ids
            arr = np.array(token_ids, dtype=dtype)      # 2) 转成 numpy 数组（更适合写二进制）
            arr.tofile(f_out)                           # 3) 直接以二进制写入 .bin 文件

            p_bar.update(len(line.encode("utf-8")))

    p_bar.close()


def load_tokenizer_from_dir(dir_path: str, special_tokens: list[str] = None) -> BPETokenizer:
    """此处的special tokens是用户提供的特殊token列表，不一定和训练时完全一样，主要是为了支持用户在训练时没有但推理时需要的特殊token（比如一些特定的控制符）。"""
    vocab_path = os.path.join(dir_path, "vocab.json")
    merges_path = os.path.join(dir_path, "merges.txt")
    special_tokens_path = os.path.join(dir_path, "special_tokens.txt")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    return tokenizer



if __name__ == "__main__":
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="../../datasets/tiny_stories/vocab.json",
        merges_filepath="../../datasets/tiny_stories/merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    print("Tokenizer loaded.")

    s1 = "Hello, world!"
    s2 = "Hello <|endoftext|> World!"

    print(f"{tokenizer.encode(s1)}")
    print(f"{tokenizer.encode(s2)}")

    print(f"{tokenizer.decode(tokenizer.encode(s1))}")
    print(f"{tokenizer.decode(tokenizer.encode(s2))}")

    print([tokenizer.vocab[token] for token in tokenizer.encode(s1)])
    print([tokenizer.vocab[token] for token in tokenizer.encode(s2)])

    import tiktoken
    print("\nGPT2:")
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    print(f"{gpt2_tokenizer.encode(s1)}")
    print(f"{gpt2_tokenizer.encode(s2, allowed_special={'<|endoftext|>'})}")

    print(f"{gpt2_tokenizer.decode(gpt2_tokenizer.encode(s1))}")
    print(f"{gpt2_tokenizer.decode(gpt2_tokenizer.encode(s2, allowed_special={'<|endoftext|>'}))}")

    print([gpt2_tokenizer.decode([token]) for token in gpt2_tokenizer.encode(s1)])
    print([gpt2_tokenizer.decode([token]) for token in gpt2_tokenizer.encode(s2, allowed_special={'<|endoftext|>'})])