import argparse
import os
import csv
import random
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from sklearn.metrics import precision_recall_fscore_support
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import ClassificationLossCompute, MultipleChoiceLossCompute


# Extract, clean and transforms values from a dataset:
def _stance(path):
    def clean_ascii(text):
        # function to remove non-ASCII chars from data
        return ''.join(i for i in text if ord(i) < 128)
    orig = pd.read_csv(path, delimiter='\t', header=0, encoding = "latin-1")
    orig['Tweet'] = orig['Tweet'].apply(clean_ascii)
    df = orig
    X = df.Tweet.values
    stances = ["AGAINST", "FAVOR", "NONE", "UNKNOWN"]
    class_nums = {s: i for i, s in enumerate(stances)}
    Y = np.array([class_nums[s] for s in df.Stance])
    return X, Y


# Extract the training, validation and testing data from the training and testing datasets:
def stance(data_dir, trainfile, testfile):
    path = Path(data_dir)

    X, Y = _stance(path/trainfile)
    teX, _ = _stance(path/testfile)
    tr_text, va_text, tr_sent, va_sent = train_test_split(X, Y, test_size=0.2, random_state=3535999445)
    trX = []
    trY = []
    for t, s in zip(tr_text, tr_sent):
        trX.append(t)
        trY.append(s)

    vaX = []
    vaY = []
    for t, s in zip(va_text, va_sent):
        vaX.append(t)
        vaY.append(s)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX, trY), (vaX, vaY), (teX, )


def transform_stance(X1):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    for i, x1 in enumerate(X1):
        x12 = [start] + x1[:max_len] + [clf_token]
        l12 = len(x12)
        xmb[i, 0, :l12, 0] = x12
        mmb[i, 0, :l12] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


# Make predictions and write them to a TSV target file:
def predict(submission_dir, filename):
    pred_fn = argmax
    label_decoder = None
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir)
			

def output_predictions(data_dir, test_file, submission_dir, pred_file, out_path):
    test_path = os.path.join(data_dir, test_file)
    pred_path = os.path.join(submission_dir, pred_file)
    test = pd.read_csv(test_path, delimiter='\t', header=0, encoding = "latin-1")
    def clean_ascii(text):
        # function to remove non-ASCII chars from data
        return ''.join(i for i in text if ord(i) < 128)
    test['Tweet'] = test['Tweet'].apply(clean_ascii)
    print(test)
    pred = pd.read_csv(pred_path, header=0, delimiter='\t')
    print(pred)
    pred['prediction'] = pred['prediction'].astype('int64')
    df = test.join(pred)
    #print(df)
    stances = ["AGAINST", "FAVOR", "NONE", "UNKNOWN"]
    df["Stance"] = df["prediction"].apply(lambda i: stances[i])
    df = df[["index", "Target", "Tweet", "Stance"]]
    class_nums = {s: i for i, s in enumerate(stances)}
    df.to_csv(out_path, sep='\t', index=False, header=['ID', 'Target', 'Tweet', 'Stance'])


def show_score_model(data_dir, gold_file, predictionsfile_path):
    count_false_negatives = 0
    count_false_positives = 0
    count_true_negatives = 0
    count_true_positives = 0
    goldfile_path = os.path.join(data_dir, gold_file)
    with open(predictionsfile_path, 'r') as fp:
        with open(goldfile_path, 'r') as fg:
            pred_lines = fp.readlines()[1:]
            gold_lines = fg.readlines()[1:]
            pred_stances = []
            gold_stances = []
            for i in range(len(pred_lines)):
                if len(pred_lines[i].split("\t")) > 3:
                    pred_stances.append(pred_lines[i].split("\t")[3])
                    gold_stances.append(gold_lines[i].split("\t")[3])
            cm = confusion_matrix(gold_stances, pred_stances)
            cm = cm / cm.astype(np.float).sum(axis=0)
            precision, recall, fscore, support = precision_recall_fscore_support(gold_stances, pred_stances)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F-score:", fscore)
            print("Support:", support)
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
            classNames = ['AGAINST','FAVOR','NONE']
            plt.title('Abotion Stance Confusion Matrix - Test Data')
            plt.ylabel('True stance')
            plt.xlabel('Predicted stance')
            tick_marks = np.arange(len(classNames))
            plt.xticks(tick_marks, classNames, rotation=45)
            plt.yticks(tick_marks, classNames)
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, str(round(cm[i][j] * 100, 1))+"%")
            plt.show()
        


argmax = lambda x: np.argmax(x, 1)

if __name__ == '__main__':
	# Parse the arguments passed to the program:
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--score_model', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    train_model  = args.train_model
    score_model  = args.score_model
    n_ctx = args.n_ctx
    save_dir = "../save/"
    desc = "stance"
    data_dir = "../datasets"
    log_dir = "../log/"
    submission_dir = "../default/"
    predictions_filename = "stance.tsv"
    best_params_filename = "best_params"
    encoder_path = "../model/encoder_bpe_40000.json"
    bpe_path = "../model/vocab_40000.bpe"
    train_file = "training/training-data.txt"
    test_file = "testing/testing-data.txt"
    gold_file = "gold/gold_file.txt"
    out_path = "../results/predicted.txt"	

    if train_model or submit:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print("device", device, "n_gpu", n_gpu)
        
        logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
        text_encoder = TextEncoder(encoder_path, bpe_path)
        encoder = text_encoder.encoder
        n_vocab = len(text_encoder.encoder)
        
        print("Encoding dataset...")
        ((trX, trY), (vaX, vaY), (teX, )) = encode_dataset(*stance(data_dir, train_file, test_file),
         encoder=text_encoder)
        
        encoder['_start_'] = len(encoder)
        encoder['_classify_'] = len(encoder)
        clf_token = encoder['_classify_']
        n_special = 2
        max_len = n_ctx - 2
        
        # Define maximum context as the minimum of [512, x] where x is the max sentence length
        n_ctx = min(max(
        [len(x[:max_len]) for x in trX]
        + [len(x[:max_len]) for x in vaX]
        + [len(x[:max_len]) for x in teX]
        ) + 3, n_ctx)
        
        vocab = n_vocab + n_special + n_ctx
        trX, trM = transform_stance(trX)
        vaX, vaM = transform_stance(vaX)
        
        if submit:
            teX, teM = transform_stance(teX)
        
        n_train = len(trY)
        n_valid = len(vaY)
        n_batch_train = args.n_batch * max(n_gpu, 1)
        n_updates_total = (n_train // n_batch_train) * args.n_iter
        
        dh_model = DoubleHeadModel(args, clf_token, ('classification', 3), vocab, n_ctx)
        
        criterion = nn.CrossEntropyLoss(reduce=False)
        model_opt = OpenAIAdam(dh_model.parameters(),
                               lr=args.lr,
                               schedule=args.lr_schedule,
                               warmup=args.lr_warmup,
                               t_total=n_updates_total,
                               b1=args.b1,
                               b2=args.b2,
                               e=args.e,
                               l2=args.l2,
                               vector_l2=args.vector_l2,
                               max_grad_norm=args.max_grad_norm)
        compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                     criterion,
                                                     args.lm_coef,
                                                     model_opt)
        load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)
        
        dh_model.to(device)
        dh_model = nn.DataParallel(dh_model)
        
        n_updates = 0
        n_epochs = 0
        trYt = trY		
		
    # Train the model:
    if train_model:
        path = os.path.join(save_dir, best_params_filename)
        torch.save(dh_model.state_dict(), make_path(path))
        best_score = 0
        for i in range(args.n_iter):
            print("running epoch", i)
            run_epoch()
            n_epochs += 1
            log(save_dir)	

    # Make predictions based on the trained model:
    if submit:
        path = os.path.join(save_dir, best_params_filename)
        torch_data = torch.load(path)
        print(torch_data)
        dh_model.load_state_dict(torch_data)
        predict(submission_dir, predictions_filename)
        output_predictions(data_dir, test_file, save_dir, predictions_filename, out_path)

    if score_model:
        show_score_model(data_dir, gold_file, out_path)	
