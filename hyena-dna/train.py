# @title GenomicBenchmark dataset
# From https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL

"""
We provide simple training code for the GenomicBenchmark datasets.

The GenomicBenchmarks dataset will automatically download to /contents on colab.
There are 8 datasets to choose from.
"""

from random import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded

import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig

# HyenaDNA
from standalone_hyenadna import *
from huggingface_wrapper import *


# helper functions
def exists(val):
    return val is not None


def coin_flip():
    return random() > 0.5


string_complement_map = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "a": "t",
    "c": "g",
    "g": "c",
    "t": "a",
}


# augmentation
def string_reverse_complement(seq):
    rev_comp = ""
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class GenomicBenchmarkDataset(torch.utils.data.Dataset):
    """
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.

    Genomic Benchmarks Dataset, from:
    https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


    """

    def __init__(
        self,
        split,
        max_length,
        dataset_name="human_enhancers_cohn",
        d_output=2,  # default binary classification
        # dest_path="/content", # default for colab
        dest_path="content",
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(
            x,
            # add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        seq = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq.append(self.tokenizer.sep_token_id)

        # convert to tensor
        seq = torch.LongTensor(seq)

        # need to wrap in list
        target = torch.LongTensor([y])

        return seq, target


def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    """Training loop."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, loss_fn):
    """Test loop."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.squeeze()).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def run_train():
    """
    Main entry point for training.  Select the dataset name and metadata, as
    well as model and training args, and you're off to the genomic races!

    ### GenomicBenchmarks Metadata
    # there are 8 datasets in this suite, choose 1 at a time, with their corresponding settings
    # name                                num_seqs        num_classes     median len    std
    # dummy_mouse_enhancers_ensembl       1210            2               2381          984.4
    # demo_coding_vs_intergenomic_seqs    100_000         2               200           0
    # demo_human_or_worm                  100_000         2               200           0
    # human_enhancers_cohn                27791           2               500           0
    # human_enhancers_ensembl             154842          2               269           122.6
    # human_ensembl_regulatory            289061          3               401           184.3
    # human_nontata_promoters             36131           2               251           0
    # human_ocr_ensembl                   174756          2               315           108.1

    """
    # experiment settings:
    num_epochs = 100  # ~100 seems fine
    max_length = 500  # max len of sequence of dataset (of what you want)
    use_padding = True
    dataset_name = "human_enhancers_cohn"
    batch_size = 256
    learning_rate = 6e-4  # good default for Hyena
    rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1

    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = (
        "hyenadna-tiny-1k-seqlen"  # use None if training from scratch
    )

    # we need these for the decoder head, if using
    use_head = True
    n_classes = 2

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ["hyenadna-tiny-1k-seqlen"]:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            "./checkpoints",
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    else:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        # add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )

    # create datasets
    ds_train = GenomicBenchmarkDataset(
        max_length=max_length,
        use_padding=use_padding,
        split="train",
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    ds_test = GenomicBenchmarkDataset(
        max_length=max_length,
        use_padding=use_padding,
        split="test",
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)

    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        test(model, device, test_loader, loss_fn)
        optimizer.step()
