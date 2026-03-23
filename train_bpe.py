import os

from cs336_basics.tokenizer.bpe_2 import train_bpe
from cs336_basics.tokenizer.tokenizer import encode_file_to_bin, load_tokenizer_from_dir

TINY_STORIES = {
    "train_data_path": "data/TinyStoriesV2-GPT4-train.txt",
    "dev_data_path": "data/TinyStoriesV2-GPT4-valid.txt",
    "vocab_size": 10_000,
    "special_tokens": ["<|endoftext|>"],
    "save_dir": "./datasets/tiny_stories",
}

OWT = {
    "train_data_path": "data/owt_train.txt",
    "dev_data_path": "data/owt_valid.txt",
    "vocab_size": 32_000,
    "special_tokens": [],
    "save_dir": "./datasets/owt",
}


if __name__ == "__main__":
    print("Starting BPE training script...", flush=True)

    # dataset = TINY_STORIES
    dataset = OWT

    if not os.path.exists(dataset["save_dir"]):
        print(f"Creating directory {dataset['save_dir']}...", flush=True)
        os.makedirs(dataset["save_dir"])
        train_bpe(
            dataset["train_data_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            desired_num_chunks=16,  # 增加到8个chunks来加速处理
            save_path=dataset["save_dir"],
            verbose=True,
        )

        print(f"BPE tokenizer trained and saved to {dataset['save_dir']}")

    elif os.path.exists(os.path.join(dataset["save_dir"], "vocab.json")) and \
        os.path.exists(os.path.join(dataset["save_dir"], "merges.txt")):
        print(f"Tokenizer already exists at {dataset['save_dir']}, skipping training.")

    else:
        print(f"Save directory {dataset['save_dir']} exists but tokenizer files not found. Training...", flush=True)
        train_bpe(
            dataset["train_data_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            desired_num_chunks=16,  # 增加到8个chunks来加速处理
            save_path=dataset["save_dir"],
            verbose=True,
        )

    # Pre-tokenize the dataset
    tokenizer = load_tokenizer_from_dir(dataset["save_dir"], special_tokens=dataset["special_tokens"])

    out_bin_path = os.path.join(dataset["save_dir"], "train.bin")
    encode_file_to_bin(tokenizer, dataset["train_data_path"], out_bin_path)
    print(f"Encoded training data saved to {out_bin_path}")

    out_bin_eval_path = os.path.join(dataset["save_dir"], "eval.bin")
    encode_file_to_bin(tokenizer, dataset["dev_data_path"], out_bin_eval_path)
    print(f"Encoded evaluation data saved to {out_bin_eval_path}")
