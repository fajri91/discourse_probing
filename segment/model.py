from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import get_model, bart_t5_forward
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np


class ModelData():
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.MAX_TOKEN = args.max_token

    def preprocess_one(self, edus):
        tokenized_edus = []
        labels = []
        for edu in edus:
            cur_tokenized_edu = self.tokenizer.tokenize(edu)
            cur_label = [0] * len(cur_tokenized_edu)
            cur_label[-1] = 1
            tokenized_edus += self.tokenizer.tokenize(edu)
            labels += cur_label
        assert len(tokenized_edus) == len(labels)

        if len(tokenized_edus) > self.MAX_TOKEN:
            tokenized_edus = tokenized_edus[:self.MAX_TOKEN]
            labels = labels[:self.MAX_TOKEN]
        tokenized_edus = self.tokenizer.convert_tokens_to_ids(tokenized_edus)
        labels[-1]=0
        return tokenized_edus, labels
    
    def preprocess(self, all_edus):
        output = []
        for edus in all_edus:
            output.append(self.preprocess_one(edus))
        return output


class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, tokenizer, data, idx, batch_size, device):
        PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        label = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        mask_src = 0 + (src != PAD_ID)
        
        self.src = src.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.label, self.mask_src


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.model = get_model(args.model_name, args.num_layers)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction='sum') 

    def forward(self, src,  mask_src):
        with torch.no_grad():
            if self.args.model_type in ['t5', 'bart']:
                top_vec  = bart_t5_forward(self.model, src, mask_src, self.args.num_layers)
            elif self.args.model_type in ['electra']:
                top_vec = self.model(input_ids=src, attention_mask=mask_src)[0]
            else:
                top_vec, _ = self.model(input_ids=src, attention_mask=mask_src)
        final_rep = self.dropout(top_vec)
        conclusion = self.linear(final_rep).squeeze()
        return self.sigmoid(conclusion) * mask_src

    def get_loss(self, batch_data):
        src, label, mask_src = batch_data
        output = self.forward(src, mask_src)
        return self.loss(output, label.float())

    def predict(self, src, mask_src, label):
        output = self.forward(src, mask_src)
        output = (output > 0.5) + 0
        prediction = self.transform(output, mask_src)
        answer = self.transform(label, mask_src)
        return answer, prediction

    def transform(self, tensor, mask_src):
        results = []
        array = tensor.data.cpu().numpy()
        m = mask_src.sum(dim=-1).data.cpu().numpy()
        for idx in range(len(array)):
            cur_arr = array[idx][:m[idx]]
            now=0
            result=[]
            for idy in range(len(cur_arr)):
                if cur_arr[idy]==1:
                    result.append((now, idy))
                    now=idy
                if idy == len(cur_arr)-1 and now!=idy:
                    result.append((now, idy))
            results.append(result)
        return results


def f1_score(p, g):
    count = 0
    for item in p:
        if item in g:
            count+=1
    return float(2.0*count/(len(g)+len(p)))

def calculate_f1_macro(preds, golds):
    f1_macro = []
    assert len(preds)==len(golds)
    for idx in range(len(preds)):
        f1_macro.append(f1_score(preds[idx], golds[idx]))
    return np.mean(f1_macro)

def prediction(tokenizer, dataset, model, args):
    preds = []
    golds = []
    model.eval()
    for j in range(0, len(dataset), args.batch_size):
        src, label, mask_src = Batch(tokenizer, dataset, j, args.batch_size, args.device).get()
        answer, prediction = model.predict(src, mask_src, label)
        golds += answer
        preds += prediction
    return calculate_f1_macro(preds, golds), preds
