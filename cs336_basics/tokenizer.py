"""Tokenizer class."""

from __future__ import annotations

import os
from typing import Optional
from collections.abc import Iterable

import regex as re
from collections import Counter, defaultdict
from itertools import chain, pairwise, batched
from functools import cache
import multiprocessing
import time
from tqdm import tqdm

from cs336_basics import token_utils, pretokenization_example, utils

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_byte_pair_counts_global(freq_table: list[tuple[str], int]) -> dict[tuple[str],int]:
        byte_pair_counts = Counter()
        for pt_word, freq in freq_table:
            for byte_pair in pairwise(pt_word):
                byte_pair_counts[byte_pair] += freq
        return byte_pair_counts

def chunk_clean_count_global(start, end, input_path) -> Counter:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        splitted = re.split('('+'|'.join([re.escape(st) for st in ["<|endoftext|>"]])+')', chunk)
        pre_tokenized_corpus = [re.findall(PAT, spl) if spl not in ["<|endoftext|>"] else [spl] for spl in splitted]
        freq_table = Counter(tuple(pt_word.encode()[i:i+1] for i in range(len(pt_word.encode())))
            if pt_word not in ["<|endoftext|>"] else tuple([pt_word.encode()])
            for pt_doc in pre_tokenized_corpus for pt_word in pt_doc
        )
        return freq_table

def chunk_clean_global(start, end, input_path) -> list[str]:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        splitted = re.split('('+'|'.join([re.escape(st) for st in ["<|endoftext|>"]])+')', chunk)
        pre_tokenized_corpus = [re.findall(PAT, spl) if spl not in ["<|endoftext|>"] else [spl] for spl in splitted]
        return pre_tokenized_corpus


class BPETokenizer:
    def __init__(self, vocab = None, merges = None, special_tokens: Optional[list[str]]=None):
        self.vocab = vocab
        self.token_to_int = None
        self.merges = merges
        self.special_tokens = sorted(special_tokens,key=len,reverse=True) if special_tokens else []
        self.num_workers = os.process_cpu_count()

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "BPETokenizer":
        vocab, merges = token_utils.load_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @cache
    def _merge(self, token: tuple[bytes], merge_pair: tuple[bytes]) -> tuple[bytes]:
        if merge_pair[0] not in token:
            return token
        new_tokens = []
        j = 0
        while j < len(token):
            if j < len(token)-1 and merge_pair == (token[j], token[j+1]):
                new_tokens.append(token[j]+token[j+1])
                j += 2
            else:
                new_tokens.append(token[j])
                j += 1
        return tuple(new_tokens)
    
    def _token_to_int(self, pre_token_enc):
        if pre_token_enc in self.token_to_int:
            return [self.token_to_int[pre_token_enc]]
        return [self.token_to_int[bytes([pte])] for pte in pre_token_enc]
    
    def encode_from_file(self, input_path, parallel: bool = True):
        if not parallel:
            with open(input_path, "rb") as f:
                text = f.read()
            return self.encode(text)
        
        self.token_to_int = {v:k for k, v in self.vocab.items()}
        with open(input_path, "rb") as f:
            boundaries = pretokenization_example.find_chunk_boundaries(f, self.num_workers, b"<|endoftext|>")
        items = [(start, end, input_path) for start, end in zip(boundaries[:-1],boundaries[1:])]
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            res = pool.starmap(chunk_clean_global, items)
        return self._encode_pre_tokenized_corpus(chain.from_iterable(res))

    def _encode_pre_tokenized_corpus(self, pre_tokenized_corpus):
        corpus_enc = []
        for pt_doc in tqdm(pre_tokenized_corpus,total=sum(1 for ptc in pre_tokenized_corpus)):
            if not pt_doc:
                continue
            doc_enc = []
            for pre_token in pt_doc:
                pre_token_enc = pre_token.encode()
                if pre_token_enc not in self.token_to_int:
                    pre_token_enc = tuple(pre_token_enc[i:i+1] for i in range(len(pre_token_enc)))
                else:
                    pre_token_enc = tuple([pre_token_enc])
                
                if len(pre_token_enc) == 1:
                    doc_enc.append(self.token_to_int[pre_token_enc[0]])
                    continue
                for merge in self.merges:
                    pre_token_enc = self._merge(pre_token_enc, merge)
                doc_enc += [self.token_to_int[pte] for pte in pre_token_enc]
            corpus_enc += doc_enc
        return corpus_enc

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        self.token_to_int = {v:k for k, v in self.vocab.items()}
        pre_tokenized_corpus = self._clean_input(text)
        return self._encode_pre_tokenized_corpus(pre_tokenized_corpus)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[str]:
        for it in iterable:
            for tokens in  self.encode(it):
                yield tokens

    def decode(self, ids: list[int]) -> str:
        text = b""
        if ids and isinstance(ids[0],list):
            ids = chain.from_iterable(ids)
        for id_ in tqdm(ids,total=len(ids)):
            text += self.vocab[id_]
        return text.decode(errors='replace')
    
    def _clean_input(self, file_content):
        if self.special_tokens:
            splitted = re.split('('+'|'.join([re.escape(st) for st in self.special_tokens])+')', file_content)
        else:
            splitted = [file_content]
        return [re.findall(PAT, spl) if spl not in self.special_tokens else [spl] for spl in splitted]
   
    @utils.stopwatch
    def _chunk_clean_count_parallel(self, input_path):
        with open(input_path, "rb") as f:
            boundaries = pretokenization_example.find_chunk_boundaries(f, self.num_workers, b"<|endoftext|>")
        items = [(start, end, input_path) for start, end in zip(boundaries[:-1],boundaries[1:])]
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            res = pool.starmap(chunk_clean_count_global, items)
        for r in res[1:]:
            res[0] += r
        return res[0]

    def _get_byte_pair_counts(self, freq_table: dict[tuple[str],int]) -> dict[tuple[str],int]:
        byte_pair_counts = defaultdict(int)
        byte_pair_words = defaultdict(list)
        for pt_word, freq in freq_table.items():
            for i, byte_pair in enumerate(pairwise(pt_word)):
                byte_pair_counts[byte_pair] += freq
                byte_pair_words[byte_pair].append((pt_word, i))
        return byte_pair_counts, byte_pair_words
    
    def _get_byte_pair_counts_parallel(self, freq_table: dict):
        items = batched(freq_table.items(),len(freq_table)//max(1,self.um_workers-1))
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            res = pool.map(get_byte_pair_counts_global, items)
        byte_pair_counts = Counter()
        for r in res:
            byte_pair_counts += r
        return byte_pair_counts

    def _update_byte_pairs(self, max_pair: tuple) -> list(tuple(bytes, int)):
        new_counts = defaultdict(int)
        new_byte = tuple([max_pair[0]+max_pair[1]],)
        for word, pos in self.byte_pair_words[max_pair]:
            if not word in self.freq_table:
                continue
            new_word = word[:pos] + new_byte + word[pos+2:]
            cnt = self.freq_table[word]
            self.freq_table[new_word] = cnt
            del self.freq_table[word]
            left, right = new_word[pos-1:pos+1], new_word[pos:pos+2]
            for i, pair in enumerate(pairwise(new_word)):
                if pair == left or pair == right:
                    continue
                self.byte_pair_words[pair].append((new_word,i))
            if len(left) == 2:
                new_counts[left] += cnt
                self.byte_pair_counts[left] += cnt
                self.byte_pair_words[left].append((new_word, pos-1))
                self.byte_pair_counts[(word[pos-1],word[pos])] -= cnt
            if len(right) == 2:
                new_counts[right] += cnt
                self.byte_pair_counts[right] += cnt
                self.byte_pair_words[right].append((new_word, pos))
                self.byte_pair_counts[(word[pos+1],word[pos+2])] -= cnt
        del self.byte_pair_words[max_pair]
        return new_counts
        
    
    def train_tokenizer(self, input_path, vocab_size, special_tokens = [], parallel: bool = True) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.i = 256
        for st in special_tokens:
            self.vocab[self.i] = st.encode('utf-8')
            self.i += 1
        self.merges = list()
        self.special_tokens = special_tokens

        if not parallel:
            start = time.time()
            with open(input_path, 'r') as f:
                file_content = f.read()
            
            pre_tokenized_corpus = self._clean_input(file_content)
            self.freq_table = Counter(tuple(pt_word.encode()[i:i+1] for i in range(len(pt_word.encode())))
                if pt_word not in self.special_tokens else tuple([pt_word.encode()])
                for pt_doc in pre_tokenized_corpus for pt_word in pt_doc
            )
            print(f"Sequential: {time.time() - start :.3f} s")
        else:
            start = time.time()
            self.freq_table = self._chunk_clean_count_parallel(input_path)
            print(f"Parallel: {time.time() - start :.3f} s")

        self.byte_pair_counts, self.byte_pair_words = self._get_byte_pair_counts(self.freq_table)
        with tqdm(total=vocab_size-len(self.vocab), desc=f"It {self.i}/{vocab_size}") as pbar:
            while self.i < vocab_size:
                max_pair, _ = max(self.byte_pair_counts.items(), key=lambda kv: (kv[1],kv[0]))
                self.merges.append((max_pair[0],max_pair[1]))
                self.vocab[self.i] = max_pair[0] + max_pair[1]
                new_byte_pair_counts = self._update_byte_pairs(max_pair)
                del self.byte_pair_counts[max_pair]
                pbar.update(1)
                self.i += 1
        return self.vocab, self.merges
