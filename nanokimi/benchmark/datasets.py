"""
Benchmark datasets for nanoKimi evaluation.

This module provides utilities for loading and preprocessing standard
benchmark datasets used in language modeling evaluation.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import urllib.request
import tarfile
import zipfile

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from ..training.data import TokenDataset, TextDataProcessor, DataConfig


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""
    
    name: str
    description: str
    size_mb: Optional[float]
    num_tokens: Optional[int]
    vocab_size: Optional[int]
    download_url: Optional[str] = None
    license: Optional[str] = None


class WikiTextDataset:
    """WikiText-103 dataset for language modeling."""
    
    def __init__(self, data_dir: str = "./data", tokenizer_name: str = "gpt2"):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_and_prepare(self, split: str = "test") -> str:
        """
        Download and prepare WikiText-103 dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            Path to processed dataset file
        """
        os.makedirs(self.data_dir, exist_ok=True)
        
        output_path = os.path.join(self.data_dir, f"wikitext103_{split}.npy")
        
        if os.path.exists(output_path):
            print(f"WikiText-103 {split} already exists at {output_path}")
            return output_path
        
        print(f"Downloading WikiText-103 {split} split...")
        
        try:
            # Load dataset using HuggingFace datasets
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            
            # Process text
            all_tokens = []
            for item in dataset:
                text = item['text'].strip()
                if text:  # Skip empty lines
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(tokens)
            
            # Save as numpy array
            token_array = np.array(all_tokens, dtype=np.uint16)
            np.save(output_path, token_array)
            
            print(f"WikiText-103 {split} saved to {output_path}")
            print(f"Total tokens: {len(all_tokens):,}")
            
            # Save metadata
            metadata = {
                'dataset': 'wikitext-103',
                'split': split,
                'num_tokens': len(all_tokens),
                'vocab_size': self.tokenizer.vocab_size,
                'tokenizer': self.tokenizer_name,
            }
            
            metadata_path = output_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download WikiText-103: {e}")
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        return DatasetInfo(
            name="WikiText-103",
            description="Large-scale language modeling dataset from Wikipedia",
            size_mb=183.0,  # Approximate compressed size
            num_tokens=103000000,  # Approximate
            vocab_size=self.tokenizer.vocab_size,
            download_url="https://huggingface.co/datasets/wikitext",
            license="Creative Commons Attribution-ShareAlike License"
        )


class PennTreebankDataset:
    """Penn Treebank dataset for language modeling."""
    
    def __init__(self, data_dir: str = "./data", tokenizer_name: str = "gpt2"):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_and_prepare(self, split: str = "test") -> str:
        """
        Download and prepare Penn Treebank dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            Path to processed dataset file
        """
        os.makedirs(self.data_dir, exist_ok=True)
        
        output_path = os.path.join(self.data_dir, f"ptb_{split}.npy")
        
        if os.path.exists(output_path):
            print(f"Penn Treebank {split} already exists at {output_path}")
            return output_path
        
        print(f"Downloading Penn Treebank {split} split...")
        
        try:
            # Load dataset using HuggingFace datasets
            dataset = load_dataset("ptb_text_only", split=split)
            
            # Process text
            all_tokens = []
            for item in dataset:
                text = item['sentence'].strip()
                if text:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(tokens)
            
            # Save as numpy array
            token_array = np.array(all_tokens, dtype=np.uint16)
            np.save(output_path, token_array)
            
            print(f"Penn Treebank {split} saved to {output_path}")
            print(f"Total tokens: {len(all_tokens):,}")
            
            # Save metadata
            metadata = {
                'dataset': 'penn-treebank',
                'split': split,
                'num_tokens': len(all_tokens),
                'vocab_size': self.tokenizer.vocab_size,
                'tokenizer': self.tokenizer_name,
            }
            
            metadata_path = output_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download Penn Treebank: {e}")
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        return DatasetInfo(
            name="Penn Treebank",
            description="Classic language modeling benchmark dataset",
            size_mb=5.1,  # Approximate compressed size
            num_tokens=1000000,  # Approximate
            vocab_size=self.tokenizer.vocab_size,
            download_url="https://huggingface.co/datasets/ptb_text_only",
            license="LDC User Agreement"
        )


class LambadaDataset:
    """LAMBADA dataset for cloze-style language modeling."""
    
    def __init__(self, data_dir: str = "./data", tokenizer_name: str = "gpt2"):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_and_prepare(self) -> str:
        """
        Download and prepare LAMBADA dataset.
        
        Returns:
            Path to processed dataset file
        """
        os.makedirs(self.data_dir, exist_ok=True)
        
        output_path = os.path.join(self.data_dir, "lambada_test.npy")
        
        if os.path.exists(output_path):
            print(f"LAMBADA test already exists at {output_path}")
            return output_path
        
        print("Downloading LAMBADA test set...")
        
        try:
            # Load dataset using HuggingFace datasets
            dataset = load_dataset("lambada", split="test")
            
            # Process text
            all_tokens = []
            for item in dataset:
                text = item['text'].strip()
                if text:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(tokens)
            
            # Save as numpy array
            token_array = np.array(all_tokens, dtype=np.uint16)
            np.save(output_path, token_array)
            
            print(f"LAMBADA test saved to {output_path}")
            print(f"Total tokens: {len(all_tokens):,}")
            
            # Save metadata
            metadata = {
                'dataset': 'lambada',
                'split': 'test',
                'num_tokens': len(all_tokens),
                'vocab_size': self.tokenizer.vocab_size,
                'tokenizer': self.tokenizer_name,
            }
            
            metadata_path = output_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download LAMBADA: {e}")
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        return DatasetInfo(
            name="LAMBADA",
            description="Language modeling benchmark for long-range dependencies",
            size_mb=2.5,  # Approximate
            num_tokens=200000,  # Approximate
            vocab_size=self.tokenizer.vocab_size,
            download_url="https://huggingface.co/datasets/lambada",
            license="Apache 2.0"
        )


class HellaSwagDataset:
    """HellaSwag dataset for commonsense reasoning."""
    
    def __init__(self, data_dir: str = "./data", tokenizer_name: str = "gpt2"):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_and_prepare(self, split: str = "validation") -> str:
        """
        Download and prepare HellaSwag dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            Path to processed dataset file
        """
        os.makedirs(self.data_dir, exist_ok=True)
        
        output_path = os.path.join(self.data_dir, f"hellaswag_{split}.npy")
        
        if os.path.exists(output_path):
            print(f"HellaSwag {split} already exists at {output_path}")
            return output_path
        
        print(f"Downloading HellaSwag {split} split...")
        
        try:
            # Load dataset using HuggingFace datasets
            dataset = load_dataset("hellaswag", split=split)
            
            # Process text (combine context and endings)
            all_tokens = []
            for item in dataset:
                # Combine context with all endings
                context = item['ctx'].strip()
                endings = item['endings']
                
                for ending in endings:
                    full_text = context + " " + ending.strip()
                    tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
                    all_tokens.extend(tokens)
            
            # Save as numpy array
            token_array = np.array(all_tokens, dtype=np.uint16)
            np.save(output_path, token_array)
            
            print(f"HellaSwag {split} saved to {output_path}")
            print(f"Total tokens: {len(all_tokens):,}")
            
            # Save metadata
            metadata = {
                'dataset': 'hellaswag',
                'split': split,
                'num_tokens': len(all_tokens),
                'vocab_size': self.tokenizer.vocab_size,
                'tokenizer': self.tokenizer_name,
            }
            
            metadata_path = output_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download HellaSwag: {e}")
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        return DatasetInfo(
            name="HellaSwag",
            description="Commonsense reasoning benchmark",
            size_mb=15.0,  # Approximate
            num_tokens=500000,  # Approximate
            vocab_size=self.tokenizer.vocab_size,
            download_url="https://huggingface.co/datasets/hellaswag",
            license="MIT"
        )


class BenchmarkDatasetManager:
    """Manager for all benchmark datasets."""
    
    def __init__(self, data_dir: str = "./data", tokenizer_name: str = "gpt2"):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        
        # Initialize dataset handlers
        self.datasets = {
            'wikitext103': WikiTextDataset(data_dir, tokenizer_name),
            'penntreebank': PennTreebankDataset(data_dir, tokenizer_name),
            'lambada': LambadaDataset(data_dir, tokenizer_name),
            'hellaswag': HellaSwagDataset(data_dir, tokenizer_name),
        }
    
    def list_available_datasets(self) -> Dict[str, DatasetInfo]:
        """List all available benchmark datasets."""
        return {name: dataset.get_info() for name, dataset in self.datasets.items()}
    
    def download_dataset(self, dataset_name: str, split: str = "test") -> str:
        """
        Download and prepare a specific dataset.
        
        Args:
            dataset_name: Name of dataset to download
            split: Dataset split to download
            
        Returns:
            Path to processed dataset file
        """
        if dataset_name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        dataset = self.datasets[dataset_name]
        
        if dataset_name == 'lambada':
            # LAMBADA only has test split
            return dataset.download_and_prepare()
        else:
            return dataset.download_and_prepare(split)
    
    def create_token_dataset(
        self, 
        dataset_name: str, 
        split: str = "test",
        block_size: int = 1024
    ) -> TokenDataset:
        """
        Create TokenDataset from benchmark dataset.
        
        Args:
            dataset_name: Name of benchmark dataset
            split: Dataset split
            block_size: Block size for sequences
            
        Returns:
            TokenDataset instance
        """
        data_path = self.download_dataset(dataset_name, split)
        
        return TokenDataset(
            data_path=data_path,
            block_size=block_size,
            split=split,
            tokenizer_name=self.tokenizer_name,
        )
    
    def download_all_datasets(self) -> Dict[str, Dict[str, str]]:
        """
        Download all available benchmark datasets.
        
        Returns:
            Dictionary mapping dataset names to split paths
        """
        all_paths = {}
        
        for dataset_name in self.datasets.keys():
            print(f"\nDownloading {dataset_name}...")
            
            if dataset_name == 'lambada':
                # LAMBADA only has test split
                path = self.download_dataset(dataset_name)
                all_paths[dataset_name] = {'test': path}
            else:
                # Download train, validation, test
                splits = {}
                for split in ['train', 'validation', 'test']:
                    try:
                        path = self.download_dataset(dataset_name, split)
                        splits[split] = path
                    except Exception as e:
                        print(f"Failed to download {dataset_name} {split}: {e}")
                
                all_paths[dataset_name] = splits
        
        return all_paths
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all downloaded datasets."""
        stats = {}
        
        for dataset_name in self.datasets.keys():
            dataset_stats = {}
            
            # Check which splits are available
            for split in ['train', 'validation', 'test']:
                if dataset_name == 'lambada' and split != 'test':
                    continue
                
                metadata_path = os.path.join(
                    self.data_dir, 
                    f"{dataset_name}_{split}_metadata.json"
                )
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    dataset_stats[split] = metadata
            
            if dataset_stats:
                stats[dataset_name] = dataset_stats
        
        return stats


def load_benchmark_dataset(
    dataset_name: str,
    split: str = "test", 
    data_dir: str = "./data",
    tokenizer_name: str = "gpt2",
    block_size: int = 1024,
    download: bool = True,
) -> TokenDataset:
    """
    Convenience function to load a benchmark dataset.
    
    Args:
        dataset_name: Name of dataset ('wikitext103', 'penntreebank', etc.)
        split: Dataset split ('train', 'validation', 'test')
        data_dir: Directory to store data
        tokenizer_name: Tokenizer to use
        block_size: Sequence block size
        download: Whether to download if not found
        
    Returns:
        TokenDataset instance
    """
    manager = BenchmarkDatasetManager(data_dir, tokenizer_name)
    
    if download:
        return manager.create_token_dataset(dataset_name, split, block_size)
    else:
        # Check if dataset exists
        expected_path = os.path.join(data_dir, f"{dataset_name}_{split}.npy")
        if not os.path.exists(expected_path):
            raise FileNotFoundError(
                f"Dataset not found at {expected_path}. Set download=True to download."
            )
        
        return TokenDataset(
            data_path=expected_path,
            block_size=block_size,
            split=split,
            tokenizer_name=tokenizer_name,
        )


def print_dataset_info():
    """Print information about all available benchmark datasets."""
    manager = BenchmarkDatasetManager()
    datasets_info = manager.list_available_datasets()
    
    print("Available Benchmark Datasets:")
    print("=" * 50)
    
    for name, info in datasets_info.items():
        print(f"\n{info.name}")
        print(f"  Description: {info.description}")
        print(f"  Size: {info.size_mb} MB (approximate)")
        print(f"  Tokens: {info.num_tokens:,} (approximate)")
        print(f"  License: {info.license}")
        print(f"  URL: {info.download_url}")


if __name__ == "__main__":
    # Print available datasets when run as script
    print_dataset_info()