"""Main training loop."""

import pathlib
import argparse
import time
import logging
logging.basicConfig(level=logging.INFO)

import functools

import torch
import torch.optim.lr_scheduler as lr_scheduler

from cs336_basics import utils, model, optimizer, data, tokenizer, nn_utils


DEVICE = utils.get_device()


def main(args):
    print("Parsed arguments:")
    print(f"Number of layers: {args.num_layers}")
    print(f"Number of heads: {args.num_heads}")
    print(f"Model dimensionality: {args.d_model}")
    print(f"Feed-forward dimensionality: {args.d_ff}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Theta: {args.theta}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Total tokens processed: {args.total_tokens_processed}")
    print(f"Gradient clipping norm: {args.gradient_clipping}")
    print(f"LR scheduling: {args.lr_scheduler}")
    print(f"Device: {DEVICE}")

    print(f"\nDEBUG: {args.debug}\n")



    TOTAL_STEP_COUNT = args.total_tokens_processed // (args.batch_size * args.context_length)
    print(f"Total step count: {TOTAL_STEP_COUNT}")

    lm = model.TransformerLM(
        vocab_size=args.vocab_size if not args.debug else 500,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta
    )
    print("No. params: ", sum(param.numel() for param in lm.parameters() if param.requires_grad))
    lm.compile()
    lm.to(DEVICE)

    opt = optimizer.AdamW(params=lm.parameters(),lr=args.learning_rate)
    
    if args.lr_scheduler:
        lr_lambda = functools.partial(optimizer.get_lr_cosine_schedule,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.learning_rate * 0.1,
            warmup_iters=(TOTAL_STEP_COUNT * 7)//25,
            cosine_cycle_iters=(TOTAL_STEP_COUNT * 21)//25
        )
        scheduler = lr_scheduler.LambdaLR(opt, lr_lambda)

    train_path, valid_path, vocab_path, merges_path = data.get_paths(args.learning_task, args.debug)
    bpe_tokenizer = tokenizer.BPETokenizer.from_files(vocab_path, merges_path)
    
    with open(train_path, 'r') as f:
        train_corpus = f.read()
    encoded_train_data = bpe_tokenizer.encode(train_corpus)
    with open(valid_path, 'r') as f:
        valid_corpus = f.read()
    encoded_valid_data = bpe_tokenizer.encode(valid_corpus)

    running_loss = 0.
    running_time = time.time()
    best_vloss = 1_000_000
    for step in range(1,TOTAL_STEP_COUNT+1):
        opt.zero_grad()
        
        in_features, targets = data.get_batch(encoded_train_data, args.batch_size, args.context_length, DEVICE)
        lm.train(True)
        pred = lm(in_features)
        loss = nn_utils.cross_entropy(pred, targets)
        loss.backward()
        grad_norm = nn_utils.get_grad_norm(lm.parameters())
        if args.gradient_clipping > 0:
            nn_utils.gradient_clipping(lm.parameters(), max_l2_norm=args.gradient_clipping)
        opt.step()

        if args.lr_scheduler:
            scheduler.step()

        running_loss += loss.item()
        if step % 100 == 0:
            end_time = time.time()
            time_per_batch = (end_time-running_time) / 100
            last_loss = running_loss / 100
            logging.info("Step: %i, Loss: %.5f, grad norm: %.5f, Avg. step time: %.3f seconds", step, last_loss, grad_norm, time_per_batch)
            running_loss = 0.
            running_time = time.time()

        if step % 1000 == 0:
            lm.eval()
            running_vloss = 0.
            with torch.no_grad():
                j = 0
                for in_features, targets in data.linear_batch(encoded_valid_data, args.batch_size, args.context_length, DEVICE):
                    pred = lm(in_features)
                    loss = nn_utils.cross_entropy(pred, targets)
                    running_vloss += loss.item()
                    j += 1
            logging.info("Val loss: %.5f", running_vloss / j)
            if running_vloss <= best_vloss:
                logging.info("Saving checkpoint.")
                best_vloss = running_vloss
                utils.save_checkpoint(lm, opt, step, f"./data/best-checkpoint.pt")
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script with various numerical arguments.")

    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers in the model.')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Number of attention heads.')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Dimensionality of the model.')
    parser.add_argument('--d_ff', type=int, default=1344,
                        help='Dimensionality of the feed-forward network.')
    parser.add_argument('--vocab_size', type=int, default=10_000,
                        help='Size of the vocabulary.')
    parser.add_argument('--context_length', type=int, default=256,
                        help='Maximum context length for the model.')
    parser.add_argument('--total_tokens_processed', type=int, default=327_680_000,
                        help='maximum number of tokens processed during the training run.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--theta', type=float, default=1e4,
                        help='Theta parameter (integer).')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimization (e.g., 1e-3).')
    parser.add_argument('--learning_task', type=data.LearningTask,
                        choices=list(data.LearningTask), default=data.LearningTask.TINY_STORIES_V2,
                        help='Which task to choose.')
    parser.add_argument('--gradient_clipping', type=float, default=0.0,
                        help='Max norm for gradient clipping.No clipping if 0.')
    parser.add_argument('--debug', action='store_true',
                        help='Debug run.')
    parser.add_argument('--lr_scheduler', action='store_true',
                        help='Use cosine learning rate scheduler.')

    args = parser.parse_args()
    main(args)