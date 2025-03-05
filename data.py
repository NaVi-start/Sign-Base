# coding: utf-8
import sys
import os
import io
import os.path
from typing import Optional

# from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator
import torch

from constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import build_vocab, Vocabulary

def make_data_iter(dataset: Dataset, batch_size: int, batch_type: str = "sentence", train: bool = False, shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter

# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch

# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)

def load_data(cfg: dict):
    data_cfg = cfg["data"]

    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the counter required for the model
    trg_size = cfg["model"]["trg_size"] + 1
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()
    
    # Source field is a tokenised version of the source words
    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)
    
    # Files field is just a raw text field
    files_field = data.RawField()

    def tokenize_features(features):
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]
    
    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
    
    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((trg_size,))*TARGET_PAD,
                               preprocessing=tokenize_features,
                               postprocessing=stack_features)
    
    # Create the Training Data, using the SignProdDataset
    train_data = SignProdDataset(path=train_path,
                                 fields=(src_field, reg_trg_field, files_field),
                                 trg_size=trg_size,
                                 skip_frames=skip_frames,
                                 filter_pred=
                                 lambda x: len(vars(x)['src'])
                                 <= max_sent_length
                                 and len(vars(x)['trg'])
                                 <= max_sent_length,
                                 state='train')
    
    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    
    # Create a target vocab just as big as the required target vector size -
    # So that len(trg_vocab) is # of joints + 1 (for the counter)
    trg_vocab = [None]*trg_size

    dev_data = SignProdDataset(path=dev_path,
                               trg_size=trg_size,
                               fields=(src_field, reg_trg_field, files_field),
                               skip_frames=skip_frames,
                               state='dev')
    
    src_field.vocab = src_vocab

    return train_data, dev_data, src_vocab, trg_vocab
    
# Main Dataset Class
class SignProdDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    def __init__(self, path, fields, trg_size, skip_frames=1, state='train', **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('file_paths', fields[2])]

        examples = []
        # Extract the parallel src, trg and file files
        src_trg_files = torch.load(path)

        i = 0
        key = list(src_trg_files.keys())
        # For Source, Target and FilePath
        for files_file in key:
            i += 1
            src_line, trg_line, files_line = src_trg_files[files_file]['text'], torch.reshape(src_trg_files[files_file]['poses_3d'],(src_trg_files[files_file]['poses_3d'].shape[0],src_trg_files[files_file]['poses_3d'].shape[1]*3)), state + '/' + files_file
            trg_frames = torch.cat((trg_line,create_register(trg_line.shape[0])), dim=1)
            
            # Create a dataset examples out of the Source, Target Frames and FilesPath
            if src_line != '' and trg_line != '':
                examples.append(data.Example.fromlist(
                    [src_line, trg_frames, files_line], fields))
            
        super(SignProdDataset, self).__init__(examples, fields, **kwargs)

def create_register(num):
    # 检查输入是否为正整数
    if not isinstance(num, int) or num <= 0:
        raise ValueError("输入 num 必须是正整数。")
    
    # 使用 torch.linspace 创建从 1/num 到 1 的一维张量，包含 num 个元素
    values = torch.linspace(1 / num, 1, num)
    
    # 将张量形状调整为 [num, 1]
    result = values.view(num, 1)
    
    return result



def load_data_test(cfg: dict):
    data_cfg = cfg["data"]

    # pt_path = data_cfg["test_pt"]
    text_path = data_cfg["test_text"]
    keys_path = data_cfg["keys_text"]
    trg_path = data_cfg["trg_text"]

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the counter required for the model
    trg_size = cfg["model"]["trg_size"] + 1
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()
    
    # Source field is a tokenised version of the source words
    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)
    
    # Files field is just a raw text field
    files_field = data.RawField()

    def tokenize_features(features):
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]
    
    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
    
    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((trg_size,))*TARGET_PAD,
                               preprocessing=tokenize_features,
                               postprocessing=stack_features)
    
    test_data = SignProdDataset_test(keys_path=keys_path, 
                                     text_path=text_path,
                                     trg_path=trg_path,
                                     trg_size=trg_size,
                                     fields=(src_field, reg_trg_field, files_field),
                                     skip_frames=skip_frames,
                                     state='test')
    
    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=test_data, vocab_file=src_vocab_file)

    # Create a target vocab just as big as the required target vector size -
    # So that len(trg_vocab) is # of joints + 1 (for the counter)
    trg_vocab = [None]*trg_size

    src_field.vocab = src_vocab

    return test_data, src_vocab, trg_vocab

# Main Dataset Class
class SignProdDataset_test(data.Dataset):
    """Defines a dataset for machine translation."""

    def __init__(self, keys_path, text_path, trg_path, fields, trg_size, skip_frames=1, state='train', **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('file_paths', fields[2])]

        examples = []
        # Extract the parallel src, trg and file files
        with io.open(text_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
                    io.open(keys_path, mode='r', encoding='utf-8') as files_file:

            i = 0
            # For Source, Target and FilePath
            for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
                i+= 1

                # Strip away the "\n" at the end of the line
                src_line, trg_line, files_line = src_line.strip(), trg_line.strip(), files_line.strip()
                
                trg_line = int(trg_line)
                trg_frames = torch.zeros(trg_line, trg_size)

                # Create a dataset examples out of the Source, Target Frames and FilesPath
                if src_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_frames, files_line], fields))

        super(SignProdDataset_test, self).__init__(examples, fields, **kwargs)
    
    


