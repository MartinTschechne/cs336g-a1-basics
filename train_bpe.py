"""Train tokenizer."""

import pathlib
import argparse
import logging

logging.basicConfig(level=logging.INFO)

from cs336_basics import utils, tokenizer, data, token_utils

# import cProfile

SPECIAL_TOKENS = ["<|endoftext|>"]

@utils.stopwatch
def train_tokenizer(tokenizer, input_path, args):
    return tokenizer.train_tokenizer(
        input_path= input_path,
        vocab_size= args.vocab_size if not args.debug else 500,
        special_tokens=SPECIAL_TOKENS
    )

@utils.stopwatch
def encode(tokenizer, text):
    return tokenizer.encode(text)

def main(args):
    print("Parsed arguments:")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Learning task: {args.learning_task}")
    print(f"Debug: {args.debug}")

    input_path, _, vocab_path, merges_path = data.get_paths(args.learning_task, args.debug)

    bpe_tokenizer = tokenizer.BPETokenizer()
    vocab, merges = train_tokenizer(bpe_tokenizer, input_path, args)

    token_utils.save_vocab_and_merges(
        vocab, merges, vocab_path=vocab_path, merges_path=merges_path
    )

    with open(input_path,'r') as f:
        text = f.read()
    _ = encode(bpe_tokenizer, text)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script with various numerical arguments.")

    parser.add_argument('--vocab_size', type=int, default=10_000,
                        help='Size of the vocabulary.')
    parser.add_argument('--learning_task', type=data.LearningTask,
                        choices=list(data.LearningTask), default=data.LearningTask.TINY_STORIES_V2,
                        help='Which task to choose.')
    parser.add_argument('--debug', action='store_true',
                        help='Debugging flag.')

    args = parser.parse_args()
    main(args)
    # from functools import partial
    # profiled_main = partial(main, args)
    # cProfile.run('profiled_main()')