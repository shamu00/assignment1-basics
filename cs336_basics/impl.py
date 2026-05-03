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
        
        # initialize output
        assert vocab_size >= 256
        num_merges = vocab_size - 256 - len(special_tokens)
        self.idx = 256
        
        # 1. read the file into memory
        raw_text: str = self.read_file(input_path)
        assert isinstance(raw_text, str), "should return str type"
        # handle special tokens 
        ids: tuple[tuple[int, ...]] = self.encode_with_special_tokens(raw_text, special_tokens)
        
        # 3. recursively do token merges until the vocab size satisfies
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats: dict[tuple[int, int], int] = {}
            for chunk_ids in ids:
                self.get_stats(chunk_ids, stats)
            # find the pair with highest count
            pair: tuple[int, int] = max(stats, key=lambda k: (stats[k], k))
            # find out the merging bytes and new token id
            new_ids: list[tuple[int, ...]] = []
            for chunk_ids in ids:
                new_ids.append(self.merge(chunk_ids, pair))
            ids = tuple[tuple[int, ...]](new_ids)
            bytes_pair: tuple[bytes, bytes] = (self.vocab[pair[0]], self.vocab[pair[1]])
            self.vocab[self.idx] = b''.join(bytes_pair)
            self.merges.append(bytes_pair)

            if self.verbose:
                print(f"{i}/{num_merges}: {pair} -> {self.idx} ({self.vocab[self.idx]} has {stats[pair]} occurences)")
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

    def merge(self, ids: tuple[int, ...], pair: tuple[int, int]) -> tuple[int, ...]:
        """
        In the list of integers, replace all consecutive occurences of pairs with the
        new integer token id
        eg. ids=[1,2,3,1,2], pair=(1,2), idx=4 -> [4,3,4]
        """
        output: list[int] = []
        i = 0
        
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                output.append(self.idx)
                i += 2
            else:
                output.append(ids[i])
                i += 1
            
        return tuple(output)

    def encode_with_special_tokens(
        self, 
        text: str,
        special_tokens: list[str],
    ) -> tuple[tuple[int, ...]]:
        ids = list[tuple[int, ...]]()
        
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
                if self.verbose:
                    print(f"found special token:{chunk}")
                # ids.append(special_token_map[chunk])
            else:
                ids.append(self.encode_ordinary(chunk))
        
        return tuple[tuple[int, ...]](ids)
    
    def encode_ordinary(self, text: str) -> tuple[int, ...]: 
        """encoding that igores any special tokens"""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = list[int]()
        
        for chunk in text_chunks:
           chunk_bytes: bytes = chunk.encode()
           ids.extend(chunk_bytes)
        
        return tuple(ids)
        
        
test_input_path = "/Users/linlin/repo/assignment1-basics/data/test.txt"

_, merges = Tokenizer(True).run_train_bpe(test_input_path, 280, ["<|endoftext|>"])
print(merges)