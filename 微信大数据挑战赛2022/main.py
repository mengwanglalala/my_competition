import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders,create_dataloaderslv2
from model import MultiModal,MultiModal_lv2
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate,evaluate_lv1,evaluate_lv2
from category_id_map import LV1_CATEGORY_ID_LIST

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    # print(predictions)
    # print(labels)
    loss = sum(losses) / len(losses)
    results = evaluate_lv1(predictions, labels)

    model.train()
    return loss, results


def validate_lv2(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    # print(predictions)
    # print(labels)
    loss = sum(losses) / len(losses)
    results = evaluate_lv2(predictions, labels)

    model.train()
    return loss, results

def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)


    # 2. build model and optimizers
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        macro_f1 = results['lv1_f1_macro']
        micro_f1 = results['lv1_f1_micro']
        if macro_f1 > best_score:
            best_score = macro_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'macro_f1': macro_f1},
                       f'{args.savedmodel_path}/model_lv1_epoch_{epoch}_macro_f1_{macro_f1}_micro_f1_{micro_f1}.bin')

def train_and_validate_subv2(args,lv1_id ='00'):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaderslv2(args,lv1_id)

    # 2. build model and optimizers
    model = MultiModal_lv2(args,lv1_id)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        loss, results = validate_lv2(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        macro_f2 = results['lv2_f1_macro']
        micro_f2 = results['lv2_f1_micro']
        
        if macro_f2 > best_score:
            best_score = macro_f2
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'macro_f2': macro_f2},
                       f'{args.savedmodel_path}/model_lv2_{lv1_id}_epoch_{epoch}_macro_f2_{macro_f2}_micro_f2_{micro_f2}.bin')

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)

    lv1_list = ['00','01','02','03','05','06','09','13','16','20','21']
    for lv1_id in lv1_list:
        print(f'-----------------------{lv1_id}---------------------------')
        train_and_validate_subv2(args,lv1_id = lv1_id)


if __name__ == '__main__':
    main()
