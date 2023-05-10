import torch
from torch.utils.data import SequentialSampler, DataLoader
import pandas as pd
from config import parse_args
from data_helper import MultiModalDataset,MultiModalDataset_lv2 ,MultiModalDataset_lv2_infe
from category_id_map import lv2id_to_category_id,lv1id_to_category_id
from model import MultiModal,MultiModal_lv2
from category_id_map import LV1_CATEGORY_ID_LIST

def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            pred_label_id = model(batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


def inference_lv1():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            pred_label_id = model(batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv1id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


def inference_lv2(id, lv1_id):
    args = parse_args()
    test_lv1_data = pd.read_csv('./data/result.csv',header=None)
    print(test_lv1_data.head())
    print(len(test_lv1_data),lv1_id)
    test_lv1_data = test_lv1_data[test_lv1_data[1] == int(lv1_id)]
    if len(test_lv1_data) == 0: return 
    test_lv1_data = list(test_lv1_data[0])
    # 1. load data
    dataset = MultiModalDataset_lv2_infe(args, args.test_annotation, args.test_zip_feats, lv1_class = lv1_id, test_mode=True,test_lv1_data = test_lv1_data)
    print(len(dataset))
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model = MultiModal_lv2(args,lv1_id)
    base_path = 'save/v2_two_step_pre/'
    ckpt = [
        'model_lv2_00_epoch_9_macro_f2_0.4337_micro_f2_0.7208.bin',
        'model_lv2_01_epoch_9_macro_f2_0.2909_micro_f2_0.6129.bin',
        'model_lv2_02_epoch_4_macro_f2_0.1148_micro_f2_0.2297.bin',
        'model_lv2_03_epoch_9_macro_f2_0.4785_micro_f2_0.6845.bin',
        'model_lv2_04_epoch_8_macro_f2_0.607_micro_f2_0.7964.bin',
        'model_lv2_05_epoch_0_macro_f2_0.1728_micro_f2_0.35.bin',
        'model_lv2_06_epoch_0_macro_f2_0.1818_micro_f2_0.375.bin',
        'model_lv2_07_epoch_9_macro_f2_0.5542_micro_f2_0.7353.bin',
        'model_lv2_08_epoch_7_macro_f2_0.8311_micro_f2_0.8219.bin',
        'model_lv2_09_epoch_9_macro_f2_0.1989_micro_f2_0.6929.bin',
        'model_lv2_10_epoch_8_macro_f2_0.8002_micro_f2_0.9281.bin',
        'model_lv2_11_epoch_6_macro_f2_0.6903_micro_f2_0.6886.bin',
        'model_lv2_12_epoch_7_macro_f2_0.7436_micro_f2_0.8309.bin',
        'model_lv2_13_epoch_8_macro_f2_0.399_micro_f2_0.7905.bin',
        'model_lv2_14_epoch_9_macro_f2_0.8051_micro_f2_0.8284.bin',
        'model_lv2_15_epoch_9_macro_f2_0.5863_micro_f2_0.8142.bin',
        'model_lv2_16_epoch_9_macro_f2_0.5605_micro_f2_0.5786.bin',
        'model_lv2_17_epoch_8_macro_f2_0.6378_micro_f2_0.8014.bin',
        'model_lv2_18_epoch_9_macro_f2_0.6418_micro_f2_0.7742.bin',
        'model_lv2_19_epoch_7_macro_f2_0.6823_micro_f2_0.7633.bin',
        'model_lv2_20_epoch_9_macro_f2_0.2582_micro_f2_0.6214.bin',
        'model_lv2_21_epoch_9_macro_f2_0.4862_micro_f2_0.7073.bin',
        'model_lv2_22_epoch_9_macro_f2_0.731_micro_f2_0.8027.bin'
    ]
    checkpoint = torch.load(base_path + ckpt[id], map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            pred_label_id = model(batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    test_output_csv = f'./data/result_{lv1_id}.csv'
    with open(test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference_lv1()
    for id, lv1_id in enumerate(LV1_CATEGORY_ID_LIST):
        print(f'---------------{lv1_id}--------------------')
        inference_lv2(id, lv1_id)
