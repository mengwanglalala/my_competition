import json
import random
import zipfile
from io import BytesIO
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id,category_id_to_lv1id


def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    print('size:',size)
    val_size = int(size * args.val_ratio)
    print('val_size:',val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    train_sampler = RandomSampler(train_dataset)#,num_samples = 100,replacement = True
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader

def create_dataloaderslv2(args, lv1_class):
    dataset = MultiModalDataset_lv2(args, args.train_annotation, args.train_zip_feats, lv1_class)
    size = len(dataset)
    print('size:',size)
    val_size = int(size * args.val_ratio)
    if val_size == 0: val_size =1
    print('val_size:',val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    train_sampler = RandomSampler(train_dataset)#,num_samples = 100,replacement = True
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader



class MultiModalDataset_lv2_infe(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 lv1_class: str = '00',
                 test_mode: bool = False,
                 test_lv1_data = [],
                 ):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.bert_ocr_seq_length = args.bert_ocr_seq_length
        self.bert_asr_seq_length = args.bert_asr_seq_length
        self.test_mode = test_mode
        self.lv1_class = lv1_class

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]
        tmp_anns = []
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
            #if test_lv1_data is not None:
            for anns in self.anns:
                if int(anns['id']) in test_lv1_data:
                    tmp_anns.append(anns)
        self.anns = tmp_anns

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str, types = 'title') -> tuple:
        '''
        {'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
                [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
                [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]), 
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
        '''
        if types == 'title':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        elif types == 'ocr':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_ocr_seq_length, padding='max_length', truncation=True)
        elif types == 'asr':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_asr_seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title tokens
        #title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        title_out = self.anns[idx]['title']
        asr_out = self.anns[idx]['asr']

        ocr_out = ''
        for dic in self.anns[idx]['ocr']:
            ocr_out += dic['text']
     
        final_str = title_out +'。' + asr_out+ '。' + ocr_out
        title_input, title_mask = self.tokenize_text(final_str)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
            # asr_input=asr_input,
            # asr_mask=asr_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data


class MultiModalDataset_lv2(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 lv1_class: str = '00',
                 test_mode: bool = False,
                 ):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.bert_ocr_seq_length = args.bert_ocr_seq_length
        self.bert_asr_seq_length = args.bert_asr_seq_length
        self.test_mode = test_mode
        self.lv1_class = lv1_class

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]
        tmp_anns = []
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
            for anns in self.anns:
                if anns['category_id'][0:2] == lv1_class:
                    tmp_anns.append(anns)
        self.anns = tmp_anns

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str, types = 'title') -> tuple:
        '''
        {'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
                [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
                [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]), 
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
        '''
        if types == 'title':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        elif types == 'ocr':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_ocr_seq_length, padding='max_length', truncation=True)
        elif types == 'asr':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_asr_seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title tokens
        title_out = self.anns[idx]['title']
        asr_out = self.anns[idx]['asr']

        ocr_out = ''
        for dic in self.anns[idx]['ocr']:
            ocr_out += dic['text']
     
        final_str = title_out +'。' + asr_out+ '。' + ocr_out
        title_input, title_mask = self.tokenize_text(final_str)
        # asr
        #asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'])
        # print('before:',self.anns[idx]['asr'])
        #asr_out = self.clean_sentence(self.anns[idx]['asr'])
        # asr_out = re.sub(r"[^\w]","", self.anns[idx]['asr'], flags=re.I)#去掉异常字符
        # asr_out = re.sub(r"[0-9]","", asr_out, flags=re.I)#去掉数字
        # asr_out = re.sub(r"[a-z]","", asr_out, flags=re.I)#去掉英文字母
        # asr_out = re.sub(r"啊","",asr_out)#去掉语气词
        # asr_out = re.sub(r"嗯","",asr_out)
        # asr_out = re.sub(r"哎","",asr_out)
        # asr_out = re.sub(r"呀","",asr_out)

        # asr_input, asr_mask = self.tokenize_text(asr_out,types='asr')
        # print('after:',asr_out)
        # print(self.anns[idx]['ocr'])
        # # ocr
        
        # ocr_out = ''
        # for dic in self.anns[idx]['ocr']:
        #     ocr_out += dic['text']
        # print('ocr:',ocr_out)
        # #ocr_out = self.clean_sentence(ocr_out)
        # ocr_out = re.sub(r"[^\w]","", ocr_out, flags=re.I)#去掉异常字符
        # ocr_out = re.sub(r"[0-9]","", ocr_out, flags=re.I)#去掉数字
        # ocr_out = re.sub(r"[a-z]","", ocr_out, flags=re.I)#去掉英文字母
        # ocr_out = re.sub(r"啊","",ocr_out)#去掉语气词
        # ocr_out = re.sub(r"嗯","",ocr_out)
        # ocr_out = re.sub(r"哎","",ocr_out)
        # ocr_out = re.sub(r"呀","",ocr_out)
        # print('ocr_afer:',ocr_out)
        # ocr_input, ocr_mask = self.tokenize_text(ocr_out,types='ocr')

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
            # asr_input=asr_input,
            # asr_mask=asr_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.bert_ocr_seq_length = args.bert_ocr_seq_length
        self.bert_asr_seq_length = args.bert_asr_seq_length
        self.test_mode = test_mode

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]
        tmp_anns = []
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
            # for anns in self.anns:
            #     if anns['category_id'][0:2] == '00':
            #         tmp_anns.append(anns)
        #self.anns = tmp_anns

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str, types = 'title') -> tuple:
        '''
        {'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
                [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
                [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]), 
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
        '''
        if types == 'title':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        elif types == 'ocr':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_ocr_seq_length, padding='max_length', truncation=True)
        elif types == 'asr':
            encoded_inputs = self.tokenizer(text, max_length=self.bert_asr_seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])
        return input_ids, mask
    def clean_sentence(sentence):
        sentence = re.sub(r"[^\w]","", sentence, flags=re.I)
        sentence = re.sub(r"啊","",sentence)
        sentence = re.sub(r"嗯","",sentence)
        return sentence
    def __getitem__(self, idx: int) -> dict:
        
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title tokens
        title_out = self.anns[idx]['title']
        asr_out = self.anns[idx]['asr']

        ocr_out = ''
        for dic in self.anns[idx]['ocr']:
            ocr_out += dic['text']
     
        final_str = title_out +'。' + asr_out+ '。' + ocr_out
        title_input, title_mask = self.tokenize_text(final_str)
        # asr
        #asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'])
        # print('before:',self.anns[idx]['asr'])
        #asr_out = self.clean_sentence(self.anns[idx]['asr'])
        # asr_out = re.sub(r"[^\w]","", self.anns[idx]['asr'], flags=re.I)#去掉异常字符
        # asr_out = re.sub(r"[0-9]","", asr_out, flags=re.I)#去掉数字
        # asr_out = re.sub(r"[a-z]","", asr_out, flags=re.I)#去掉英文字母
        # asr_out = re.sub(r"啊","",asr_out)#去掉语气词
        # asr_out = re.sub(r"嗯","",asr_out)
        # asr_out = re.sub(r"哎","",asr_out)
        # asr_out = re.sub(r"呀","",asr_out)

        # asr_input, asr_mask = self.tokenize_text(asr_out,types='asr')
        # print('after:',asr_out)
        # print(self.anns[idx]['ocr'])
        # # ocr
        
        # ocr_out = ''
        # for dic in self.anns[idx]['ocr']:
        #     ocr_out += dic['text']
        # print('ocr:',ocr_out)
        # #ocr_out = self.clean_sentence(ocr_out)
        # ocr_out = re.sub(r"[^\w]","", ocr_out, flags=re.I)#去掉异常字符
        # ocr_out = re.sub(r"[0-9]","", ocr_out, flags=re.I)#去掉数字
        # ocr_out = re.sub(r"[a-z]","", ocr_out, flags=re.I)#去掉英文字母
        # ocr_out = re.sub(r"啊","",ocr_out)#去掉语气词
        # ocr_out = re.sub(r"嗯","",ocr_out)
        # ocr_out = re.sub(r"哎","",ocr_out)
        # ocr_out = re.sub(r"呀","",ocr_out)
        # print('ocr_afer:',ocr_out)
        # ocr_input, ocr_mask = self.tokenize_text(ocr_out,types='ocr')

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
            # asr_input=asr_input,
            # asr_mask=asr_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv1id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
