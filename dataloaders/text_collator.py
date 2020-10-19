import torch
from torch.autograd import Variable

class TextCollator(object):
    """
    - Custom collate function for loading text data into pytorch dataloader
    - Arguments:
            + dataset:                      Text dataset
            + vocab:                        Vocabulary built from dataset   
            + include_lengths(bool):        Whether to return lengths of each sample to dataloader
            + batch_first(bool):            Whether to reshape output tensors that first dimension is batch size
            + sort_within_batch(bool):      Whether to sort samples by decreasing its lengths
            + device:                       cuda or not
            + add_init_eos(bool):           Whether to append special tokens to samples
    - Example:
            mycollate = TextCollator(dataset, vocab)
            dataloader = torch.utils.data.Dataloader(
                dataset,
                batch_size = 32,
                collate_fn = mycollate)

    """
    def __init__(self, dataset, vocab, include_lengths = False, batch_first = False, sort_within_batch = False, device = None, add_init_eos = False):
        self.train = dataset.train
        self.add_init_eos = add_init_eos
        self.dataset = dataset
        self.vocab = vocab
        self.device = device
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.sort_within_batch = sort_within_batch

    def convert_toks_to_idxs(self, sample, max_len):
        """
        Convert tokens to indexes by using vocab
        """

        # Append special tokens
        if self.add_init_eos:
            init_idx = self.vocab.stoi[self.vocab.init_token]
            eos_idx = self.vocab.stoi[self.vocab.eos_token]
        pad_idx = self.vocab.stoi[self.vocab.pad_token]
        
        # Convert text to indexes
        tokens, targets = sample["txt"], sample["label"]
        indexes = []
        for tok in tokens:
            indexes.append(self.vocab.stoi[tok])
        if self.add_init_eos:
            indexes = [init_idx] + indexes + [eos_idx]
        
        # Sentence length
        length = len(indexes)

        #Padding
        while len(indexes) < max_len:
            indexes.append(pad_idx)

        # Convert label to indexes
        target = self.dataset.classes_idx[targets]
        
        results = {"txt" : indexes,
                    "label": target}
        
        # Include lengths
        if self.include_lengths:
            results["lengths"] = length
        return results

    def sort_batch(self, idx_batch):
        """
        Sort batch by length
        """
        assert self.include_lengths , "must include lengths"
        data = [item["txt"] for item in idx_batch]
        target = [item["label"] for item in idx_batch]
        lengths = [item["lengths"] for item in idx_batch]
        # Sorting 
        sorted_batch = [[x,y,z] for x,y,z in sorted(zip(lengths ,data , target), reverse= True)]
        new_idx_batch = []
        for i in sorted_batch:
            leng, txt, target = i
            new_idx_batch.append(
                {
                    "txt": txt,
                    "lengths" : leng,
                    "label" : target
                }
            )
        return new_idx_batch

    def __call__(self, batch):
        """
        Make a batch, by making all samples equal length
        """    
        # Get length of the longest sample for padding
        max_len = 0
        for i in batch:
            max_len = max(len(i["txt"]), max_len)

        if self.add_init_eos:
            max_len += 2 #add 2 sos, eos tokens

        # Convert to indexes
        idx_batch = []
        for i in batch:
            indexes = self.convert_toks_to_idxs(i, max_len)
            idx_batch.append(indexes)

        # Sort by length
        if self.sort_within_batch:
            idx_batch = self.sort_batch(idx_batch)

        # Convert to pytorch tensor
        data = [item["txt"] for item in idx_batch]
        target = [item["label"] for item in idx_batch]
        if self.include_lengths:
            leng = [item["lengths"] for item in idx_batch]
            length = Variable(torch.LongTensor(leng))
            length = length.to(self.device) if self.device else length   
        data = Variable(torch.LongTensor(data))
        target = Variable(torch.LongTensor(target))

        # Make batch size first dimmension
        if self.batch_first:
            data = data.permute(1,0)

        # Using GPU
        if self.device is not None:
            data = data.to(self.device)
            target = target.to(self.device)

        results = {
            "txt" : data,
            "label": target }

        if self.include_lengths:
            results["len"] =  length

        return results