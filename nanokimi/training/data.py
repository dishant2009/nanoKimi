"""
Data loading and processing utilities for nanoKimi.

This module handles tokenization, dataset creation, and data loading
for training the Kimi-K2 model.
"""

import os
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    """Configuration for data processing."""
    tokenizer_name: str = "gpt2"
    block_size: int = 1024
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    preprocessing_num_workers: int = 4
    streaming: bool = False


class TokenDataset(Dataset):
    """
    Dataset for tokenized text data.
    
    Loads pre-tokenized data and serves sequences of specified length.
    """
    
    def __init__(
        self,
        data_path: str,
        block_size: int = 1024,
        split: str = "train",
        tokenizer_name: str = "gpt2",
    ):
        self.data_path = data_path
        self.block_size = block_size
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> np.ndarray:
        """Load tokenized data from file."""
        if self.data_path.endswith('.bin'):
            # Binary format (numpy array)
            data = np.fromfile(self.data_path, dtype=np.uint16)
        elif self.data_path.endswith('.npy'):
            # Numpy format
            data = np.load(self.data_path)
        elif self.data_path.endswith('.pkl'):
            # Pickle format
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
        
        return data.astype(np.int64)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data) // self.block_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1  # +1 for labels
        
        # Ensure we don't exceed data length
        if end_idx > len(self.data):
            # Wrap around or pad
            sequence = np.concatenate([
                self.data[start_idx:],
                self.data[:end_idx - len(self.data)]
            ])
        else:
            sequence = self.data[start_idx:end_idx]
        
        # Convert to tensors
        input_ids = torch.from_numpy(sequence[:-1].copy()).long()
        labels = torch.from_numpy(sequence[1:].copy()).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingTokenDataset(IterableDataset):
    """
    Streaming dataset for very large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        data_files: List[str],
        block_size: int = 1024,
        tokenizer_name: str = "gpt2",
        buffer_size: int = 1000000,
    ):
        self.data_files = data_files
        self.block_size = block_size
        self.buffer_size = buffer_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over data files and yield samples."""
        for data_file in self.data_files:
            yield from self._iter_file(data_file)
    
    def _iter_file(self, data_file: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over a single data file."""
        if data_file.endswith('.bin'):
            data = np.fromfile(data_file, dtype=np.uint16)
        elif data_file.endswith('.npy'):
            data = np.load(data_file)
        else:
            raise ValueError(f"Unsupported format: {data_file}")
        
        # Yield samples
        for i in range(0, len(data) - self.block_size, self.block_size):
            sequence = data[i:i + self.block_size + 1]
            
            input_ids = torch.from_numpy(sequence[:-1].copy()).long()
            labels = torch.from_numpy(sequence[1:].copy()).long()
            
            yield {
                'input_ids': input_ids,
                'labels': labels,
            }


class TextDataProcessor:
    """
    Process raw text data into tokenized format suitable for training.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create directories
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
    
    def process_text_file(
        self,
        input_path: str,
        output_path: str,
        chunk_size: int = 1000000,
    ) -> Dict[str, int]:
        """
        Process a text file and save tokenized data.
        
        Args:
            input_path: Path to input text file
            output_path: Path to save tokenized data
            chunk_size: Number of characters to process at once
            
        Returns:
            Statistics about processed data
        """
        print(f"Processing {input_path}...")
        
        all_tokens = []
        total_chars = 0
        total_tokens = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Tokenize chunk
                tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
                all_tokens.extend(tokens)
                
                total_chars += len(chunk)
                total_tokens += len(tokens)
                
                print(f"Processed {total_chars:,} characters, {total_tokens:,} tokens")
        
        # Convert to numpy array and save
        token_array = np.array(all_tokens, dtype=np.uint16)
        np.save(output_path, token_array)
        
        # Save metadata
        metadata = {
            'num_tokens': len(all_tokens),
            'num_chars': total_chars,
            'vocab_size': self.tokenizer.vocab_size,
            'tokenizer': self.config.tokenizer_name,
            'compression_ratio': total_chars / len(all_tokens),
        }
        
        metadata_path = output_path.replace('.npy', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(all_tokens):,} tokens to {output_path}")
        print(f"Compression ratio: {metadata['compression_ratio']:.2f} chars/token")
        
        return metadata
    
    def process_dataset(
        self,
        dataset_name: str,
        train_files: List[str],
        val_files: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Process multiple files into train/val datasets.
        
        Args:
            dataset_name: Name of the dataset
            train_files: List of training text files
            val_files: Optional list of validation text files
            
        Returns:
            Tuple of (train_data_path, val_data_path)
        """
        train_output = os.path.join(self.config.data_dir, f"{dataset_name}_train.npy")
        val_output = None
        
        # Process training files
        if len(train_files) == 1:
            self.process_text_file(train_files[0], train_output)
        else:
            self._merge_files(train_files, train_output)
        
        # Process validation files
        if val_files:
            val_output = os.path.join(self.config.data_dir, f"{dataset_name}_val.npy")
            if len(val_files) == 1:
                self.process_text_file(val_files[0], val_output)
            else:
                self._merge_files(val_files, val_output)
        
        return train_output, val_output
    
    def _merge_files(self, input_files: List[str], output_path: str):
        """Merge multiple text files and tokenize."""
        print(f"Merging {len(input_files)} files...")
        
        all_tokens = []
        
        for file_path in input_files:
            print(f"Processing {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
        
        # Save merged tokens
        token_array = np.array(all_tokens, dtype=np.uint16)
        np.save(output_path, token_array)
        
        print(f"Merged {len(all_tokens):,} tokens to {output_path}")


def load_openwebtext(data_dir: str = "./data") -> Tuple[str, str]:
    """
    Load OpenWebText dataset.
    
    This assumes you have downloaded the OpenWebText dataset.
    You can get it from: https://github.com/jcpeterson/openwebtext
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        Tuple of (train_data_path, val_data_path)
    """
    train_path = os.path.join(data_dir, "openwebtext_train.npy")
    val_path = os.path.join(data_dir, "openwebtext_val.npy")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"OpenWebText training data not found at {train_path}. "
            "Please download and preprocess the OpenWebText dataset."
        )
    
    return train_path, val_path


def load_toy_dataset(
    data_dir: str = "./data",
    tokenizer_name: str = "gpt2",
    vocab_size: int = 1000,
    seq_len: int = 10000,
) -> Tuple[str, str]:
    """
    Create a toy dataset for testing.
    
    Args:
        data_dir: Directory to save data
        tokenizer_name: Tokenizer to use
        vocab_size: Vocabulary size for toy data
        seq_len: Length of generated sequence
        
    Returns:
        Tuple of (train_data_path, val_data_path)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate random tokens
    train_tokens = np.random.randint(0, vocab_size, size=seq_len, dtype=np.uint16)
    val_tokens = np.random.randint(0, vocab_size, size=seq_len // 10, dtype=np.uint16)
    
    # Save data
    train_path = os.path.join(data_dir, "toy_train.npy")
    val_path = os.path.join(data_dir, "toy_val.npy")
    
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)
    
    print(f"Created toy dataset: {seq_len} train tokens, {seq_len // 10} val tokens")
    
    return train_path, val_path


def create_datasets(
    config: DataConfig,
    dataset_name: str = "toy",
) -> Tuple[TokenDataset, TokenDataset]:
    """
    Create train and validation datasets.
    
    Args:
        config: Data configuration
        dataset_name: Name of dataset to load
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if dataset_name == "toy":
        train_path, val_path = load_toy_dataset(
            config.data_dir,
            config.tokenizer_name,
        )
    elif dataset_name == "openwebtext":
        train_path, val_path = load_openwebtext(config.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create datasets
    train_dataset = TokenDataset(
        train_path,
        block_size=config.block_size,
        split="train",
        tokenizer_name=config.tokenizer_name,
    )
    
    val_dataset = TokenDataset(
        val_path,
        block_size=config.block_size,
        split="val",
        tokenizer_name=config.tokenizer_name,
    )
    
    return train_dataset, val_dataset


def get_batch(
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Get a random batch from dataset.
    
    Args:
        dataset: Dataset to sample from
        batch_size: Batch size
        device: Device to place tensors on
        
    Returns:
        Batch dictionary
    """
    indices = torch.randint(0, len(dataset), (batch_size,))
    
    batch_input_ids = []
    batch_labels = []
    
    for idx in indices:
        sample = dataset[idx]
        batch_input_ids.append(sample['input_ids'])
        batch_labels.append(sample['labels'])
    
    return {
        'input_ids': torch.stack(batch_input_ids).to(device),
        'labels': torch.stack(batch_labels).to(device),
    }