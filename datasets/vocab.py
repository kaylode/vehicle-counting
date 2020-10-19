import torch
import torch.utils.data as data
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import csv
from tqdm import tqdm
from utils.text_tokenizer import TextTokenizer
from collections import defaultdict
import sys
sys.path.append("..")

class CustomVocabulary(data.Dataset):
    def __init__(self,
                tokenizer = None,
                min_freqs = None,
                max_size = None,
                init_token = "<sos>",
                eos_token = "<eos>",
                pad_token = "<pad>",
                unk_token = "<unk>"):
        
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        if tokenizer is None:
            self.tokenizer = TextTokenizer()
        else:
            self.tokenizer = tokenizer
        
        self.max_size = max_size
        self.min_freqs = min_freqs
        self.freqs = {}
    
        self.vocab_size = 4
        self.special_tokens = {
            "init_token": init_token,
            "eos_token" : eos_token,
            "pad_token" : pad_token,
            "unk_token" : unk_token
        }

        self.stoi = defaultdict(lambda : 3)
        self.stoi[pad_token] = 0
        self.stoi[init_token] = 1
        self.stoi[eos_token] = 2
        self.stoi[unk_token] = 3
        
        self.itos = {
            0: pad_token,
            1: init_token,
            2: eos_token,
            3: unk_token
        }    
    
    def reset(self):
        self.vocab_size =  4
        
        self.stoi = defaultdict(lambda : 3)
        self.stoi[self.pad_token] = 0
        self.stoi[self.init_token] = 1
        self.stoi[self.eos_token] = 2
        self.stoi[self.unk_token] = 3
        
        self.itos = {
            0: self.pad_token,
            1: self.init_token,
            2: self.eos_token,
            3: self.unk_token
        } 

    def build_vocab(self, datasets):
        """
        - Build vocabulary from list of datasets
        - Argument: 
                    + datasets:     list of datasets
        """
        if not isinstance(datasets, list):
            datasets = [datasets]
        print("Building vocabulary...")
        for dataset in datasets:
            self.fns = dataset.fns
            for sentence,_ in tqdm(self.fns):
                for token in self.tokenizer.tokenize(sentence):
                    if token not in self.stoi:
                        self.stoi[token] = self.vocab_size     #index
                        self.itos[self.vocab_size] = token
                        self.vocab_size += 1
                        self.freqs[token] = 1
                    else:
                        self.freqs[token] +=1
            self.freqs = {k: v for k, v in sorted(self.freqs.items(), key=lambda item: item[1], reverse = True)}
        
        # Reduce vocabulary only contains tokens with min freqs
        if self.min_freqs is not None and self.min_freqs > 1:
            self.reset()
            new_freqs = {}
            list_freqs = list(self.freqs.items())
            for token, freqs in list_freqs:
                if freqs >= self.min_freqs:
                    new_freqs[token] = freqs
                    self.stoi[token] = self.vocab_size     #index
                    self.itos[self.vocab_size] = token
                    self.vocab_size += 1
            self.freqs = new_freqs
    
        # Reduce vocabulary size to max size
        if self.max_size is not None and self.max_size< self.vocab_size:
            self.max_size = min(self.max_size, self.vocab_size)
            self.reset()
            
            new_freqs = {}
            list_freqs = list(self.freqs.items())
            for token, freq in list_freqs:
                if self.vocab_size >= self.max_size:
                    break
                new_freqs[token] = freq
                self.stoi[token] = self.vocab_size     #index
                self.itos[self.vocab_size] = token
                self.vocab_size += 1
            self.freqs = new_freqs    
                    
        print("Vocabulary built!")
        
    def most_common(self, topk = None, ngrams = None):
        """
        Return a dict of most common words
        
        Args:
            topk: Top K words
            ngrams: string
                '1grams': unigram
                '2grams': bigrams
                '3grams': trigrams
                
        """
        
        if topk is None:
            topk = self.max_size
        idx = 0
        common_dict = {}
        
        if ngrams is None:
            for token, freq in self.freqs.items():
                if idx >= topk:
                    break
                common_dict[token] = freq
                idx += 1
        else:
            if ngrams == "1gram":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 1:
                        common_dict[token] = freq
                        idx += 1
            if ngrams == "2grams":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 2:
                        common_dict[token] = freq
                        idx += 1
            if ngrams == "3grams":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 3:
                        common_dict[token] = freq
                        idx += 1
                
            
        return common_dict
    

    def plot(self, types = None, topk = 100, figsize = (8,8) ):
        """
        Plot distribution of tokens:
            types: list
                "freqs": Tokens distribution
                "allgrams": Plot every grams
                "1gram - 2grams - 3grams" : Plot n-grams
        """
        ax = plt.figure(figsize = figsize)
        if types is None:
            types = ["freqs", "allgrams"]
        
        if "freqs" in types:
            if "allgrams" in types:
                plt.title("Top " + str(topk) + " highest frequency tokens")
                plt.xlabel("Unique tokens")
                plt.ylabel("Frequencies")
                cnt_dict = self.most_common(topk)
                bar1 = plt.barh(list(cnt_dict.keys()),
                                list(cnt_dict.values()),
                                color="blue")
            else:
                if "1gram" in types:
                    plt.title("Top " + str(topk) + " highest frequency unigram tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "1gram")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="blue",
                                    label = "Unigrams")

                if "2grams" in types:
      
                    plt.title("Top " + str(topk) + " highest frequency bigrams tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "2grams")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="gray",
                                    label = "Bigrams")

                if "3grams" in types:
                    plt.title("Top " + str(topk) + " highest frequency trigrams tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "3grams")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="green",
                                    label = "Trigrams") 
            
        plt.legend()
        plt.show()
    
    def save(self, path):
        import dill
        output = open(path, "wb")
        dill.dump(self, output)
        output.close()

    def load(self, path):
        import dill
        output = dill.load(open(path,'rb'))
        self = output
        
    def __len__(self):
        return self.vocab_size
        
    def __str__(self):
        s = "Custom Vocabulary  \n"
        line = "-------------------------------\n"
        s1 = "Number of unique words in dataset: " + str(self.vocab_size) + '\n'
        return s + line + s1 