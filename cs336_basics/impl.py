import os
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, verbose: bool=False): 
        self.verbose: bool = verbose
        self.idx: int = 0
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []
        self.compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)

    def run_train_bpe(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
        
        # reset attributes before training
        self.__init__(self.verbose)
        
        # initialize output
        special_token_set = set(special_tokens)
        assert vocab_size >= 256 + len(special_token_set)
        num_merges = vocab_size - 256 - len(special_token_set)
        self.idx = 256
        
        # 1. read the file into memory
        raw_text: str = self.read_file(input_path)
        assert isinstance(raw_text, str), "should return str type"
        # handle special tokens 
        ids: list[list[int]] = self.encode_with_special_tokens(raw_text, special_token_set)
        
        # 3. recursively do token merges until the vocab size satisfies
        # initiaze the stats, count up the number of times every consecutive pair appears
        stats: dict[tuple[int, int], int] = {}
        for chunk_ids in ids:
            self.get_stats(tuple(chunk_ids), stats)
        for i in range(num_merges):
            new_token_id: int = self.idx
            if len(stats) == 0: 
                break
            # find the pair with highest count
            pair: tuple[int, int] = max(stats, key=lambda k: (stats[k], self.vocab[k[0]], self.vocab[k[1]]))
            if stats[pair] == 0:
                break
            # find out the merging bytes and new token id
            new_ids: list[list[int]] = []
            occurrence: int = stats[pair]
            for chunk_ids in ids:
                new_ids.append(self.merge(tuple(chunk_ids), pair, stats, new_token_id))
            ids = list[list[int]](new_ids)
            bytes_pair: tuple[bytes, bytes] = (self.vocab[pair[0]], self.vocab[pair[1]])
            self.vocab[new_token_id] = b''.join(bytes_pair)
            self.merges.append(bytes_pair)

            if self.verbose:
                print(f"{i + 1}/{num_merges}: {pair} -> {self.idx} ({self.vocab[self.idx]} has {occurrence} occurrences)")
            self.idx += 1
            
        return (self.vocab, self.merges)


    def read_file(self, path: str | os.PathLike) -> str:
        with open(path) as f:
            return f.read()
        
    def get_stats(self, ids: tuple[int, ...], stats: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
        """
        given a list of integers, return a dict of counts of consecutive pairs
        eg. ids=[1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1 }
        """
        counts: dict[tuple[int, int], int] = {} if stats is None else stats
        
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids: tuple[int, ...], pair: tuple[int, int], stats: dict[tuple[int, int], int], new_token_id: int) -> list[int]:
        """
        In the list of integers, replace all consecutive occurences of pairs with the
        new integer token id
        eg. ids=[1,2,3,1,2], pair=(1,2), idx=4 -> [4,3,4]
        
        adjust the stats rather than recount it in every loop
        """
        new_ids: list[int] = []
        i = 0
        
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_token_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        i = -1
        while i + 1 < len(new_ids):
            i += 1
            if new_ids[i] != new_token_id:
                continue
            stats[(pair[0], pair[1])] -= 1
            x = new_ids[i - 1] if i > 0 else -1
            y = new_ids[i + 1] if i +1 < len(new_ids) else -1
            if x == new_token_id:
                # AB AB Y, (B, A) --
                stats[(pair[1], pair[0])] -= 1
                # (AB, AB) ++
                stats[(new_token_id, new_token_id)] = stats.get((new_token_id, new_token_id), 0) + 1
            else: 
                # X AB, (X, A) --
                if stats.get((x, pair[0]), 0) > 0:
                    stats[x, pair[0]] -= 1
                # (X, AB) ++
                if x >= 0:
                    stats[(x, new_token_id)] = stats.get((x, new_token_id), 0) + 1
            
            # if y == new_token_id, then y will be as the next x and be handled
            if y != new_token_id:
                # AB, Y, (B, Y) --
                if stats.get((pair[1], y), 0) > 0:
                    stats[(pair[1], y)] -= 1
                # (AB, Y) ++
                if y >= 0:
                    stats[(new_token_id, y)] = stats.get((new_token_id, y), 0) + 1
        
        return new_ids

    def encode_with_special_tokens(
        self, 
        text: str,
        special_tokens: set[str],
    ) -> list[list[int]]:
        ids = list[list[int]]()
        
        # tokenize special tokens
        special_token_map: dict[str, int] = {k: 0 for k in special_tokens}
        for k in special_token_map.keys():
            special_token_map[k] = self.idx
            self.vocab[self.idx] = k.encode()
            self.idx += 1

        # split up the whole text into parts by special tokens
        special_pattern = "(" + "|".join((re.escape(t) for t in special_tokens)) + ")" if len(special_tokens) > 0 else "(.+)"
        chunks = re.split(special_pattern, text)
        
        for chunk in chunks:
            if chunk in special_token_map:
                # ids.append(special_token_map[chunk])
                continue
            else:
                ids.extend(self.encode_ordinary(chunk))
        
        return ids
    
    def encode_ordinary(self, text: str) -> list[list[int]]: 
        """encoding that igores any special tokens"""
        text_chunks = re.findall(self.compiled_pattern, text)
        return [list(ch.encode()) for ch in text_chunks]

# test_input_path = "/Users/linlin/repo/assignment1-basics/data/test.txt"
# t = Tokenizer(True)
# _, merges = t.run_train_bpe(test_input_path, 258, ["<|endoftext|>"])
# print(merges)