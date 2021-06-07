import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import model2hugmodel, model2layer

logger = logging.getLogger(__name__)

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--max_token_prem', default=450, help='maximum token for four sentences combined')
args_parser.add_argument('--max_token_next', default=50, help='maximum token for one sentence')
args_parser.add_argument('--batch_size', type=int, default=40, help='batch size')
args_parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
args_parser.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
args_parser.add_argument('--seed', type=int, default=2020)
args_parser.add_argument('--local_rank', type=int, default=-1)
args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
args_parser.add_argument('--no_cuda', default=False)
args_parser.add_argument('--model_type', type=str, default='bert', \
        choices=['bert', 'roberta', 'albert', 'electra', 'gpt2', 'bart', 't5', 'bert-large', 'bert-zh', 'bert-es', 'bert-de'], help='select one of language')
args_parser.add_argument('--num_layers', type=int, default=-1, help='start from number of layers')
args_parser.add_argument('--output_folder', type=str, default='output', help='output_folder')
args_parser.add_argument('--train_data', type=str, default='data/train.csv', help='path to train data')
args_parser.add_argument('--dev_data', type=str, default='data/dev.csv', help='path to dev data')
args_parser.add_argument('--test_data', type=str, default='data/test.csv', help='path to test data')
args_parser.add_argument('--start', type=int, default=1)


# Map to huggingface model
args = args_parser.parse_args()
args.model_name = model2hugmodel[args.model_type]
if args.num_layers == -1:
    args.num_layers = model2layer[args.model_type]


# Load the right model class and methods
if args.model_type in ['bert', 'albert', 'bert-large', 'bert-zh', 'bert-es', 'bert-de']:
    from model_cls import ModelData, Batch, Model, prediction
else:
    from model_avg import ModelData, Batch, Model, prediction


# Function for reading data
def read_data(fname):
    contexts = []
    endings = []
    labels = []
    data = pd.read_csv(fname)
    for idx, row in data.iterrows():
        context = row['InputSentence1'] +' '+ row['InputSentence2'] +' '+ row['InputSentence3'] +' '+ row['InputSentence4']
        ending1 = row['RandomFifthSentenceQuiz1']
        ending2 = row['RandomFifthSentenceQuiz2']
        if row['AnswerRightEnding'] == 1:
            contexts.append(context)
            endings.append(ending1)
            labels.append(1)
            contexts.append(context)
            endings.append(ending2)
            labels.append(0)
        else:
            assert row['AnswerRightEnding'] == 2
            contexts.append(context)
            endings.append(ending1)
            labels.append(0)
            contexts.append(context)
            endings.append(ending2)
            labels.append(1)
    return contexts, endings, labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# Function for train
def train(args, train_dataset, dev_dataset, test_dataset, model):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(train_dataset) // args.batch_size * args.num_train_epochs
    warmup_steps = int(0.1 * t_total)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  LAYERS = %d", args.num_layers)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warming up = %d", warmup_steps)
    logger.info("  Patience  = %d", args.patience)

    # Added seed here for reproductibility
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    set_seed(args)
    tr_loss = 0.0
    global_step = 1
    best_acc_dev = 0
    best_acc_test = 0
    cur_patience = 0
    dev_pred = None; test_pred = None
    for i in range(int(args.num_train_epochs)):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        for j in range(0, len(train_dataset), args.batch_size):
            batch_data = Batch(tokenizer, train_dataset, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(batch_data)
            loss = loss.sum()/args.batch_size
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
        
        logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/global_step)
        dev_acc, pred = prediction(tokenizer, dev_dataset, model, args)
        if dev_acc > best_acc_dev:
            best_acc_dev = dev_acc
            dev_pred = pred
            test_acc, test_pred = prediction(tokenizer, test_dataset, model, args)
            best_acc_test = test_acc
            cur_patience = 0
            logger.info("Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)
        else:
            cur_patience += 1
            if cur_patience == args.patience:
                logger.info("Early Stopping Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)
                break
            else:
                logger.info("Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)

    return global_step, tr_loss / global_step, best_acc_dev, best_acc_test, dev_pred, test_pred


# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

modeldata = ModelData(args)

trainset = read_data(args.train_data)
devset = read_data(args.dev_data)
testset = read_data(args.test_data)
train_dataset = modeldata.preprocess(trainset[0], trainset[1], trainset[2])
dev_dataset = modeldata.preprocess(devset[0], devset[1], devset[2])
test_dataset = modeldata.preprocess(testset[0], testset[1], testset[2])

os.makedirs(args.output_folder, exist_ok = True)
output_file = args.output_folder+'/'+args.model_type+'.csv'
if args.start == 1:
    f = open(output_file, 'w')
    f.write('layers, dev, test, dev_pred, test_pred\n')
    f.close()
for idx in range(args.start, args.num_layers+1):
    args.num_layers = idx
    model = Model(args, device)
    model.to(args.device)
    global_step, tr_loss, best_acc_dev, best_acc_test, dev_pred, test_pred = train(args, train_dataset, dev_dataset, test_dataset, model)
    print('Dev set Acc', best_acc_dev)
    print('Test set Acc', best_acc_test)
    f = open(output_file, 'a+')
    str_dev_pred  = '|'.join([str(a) for a in dev_pred])
    str_test_pred = '|'.join([str(a) for a in test_pred])
    f.write(f"{idx}, {best_acc_dev}, {best_acc_test}, {str_dev_pred}, {str_test_pred}\n")
    f.close()



