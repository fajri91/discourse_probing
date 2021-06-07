from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import model2hugmodel, get_model, bart_t5_forward
from torch.autograd import Variable
import torch.nn as nn
import torch


class ModelData():
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_vid = self.tokenizer.cls_token_id
        self.sep_vid = self.tokenizer.sep_token_id
        self.pad_vid = self.tokenizer.pad_token_id
        self.MAX_TOKEN = args.max_token_word

    def preprocess_one(self, sentence, label):
        s1, s2 = sentence[0], sentence[1]
        subtoken1 = [self.cls_token] + self.tokenizer.tokenize(s1) + [self.sep_token]
        subtoken1_idxs = self.tokenizer.convert_tokens_to_ids(subtoken1)
        if len(subtoken1_idxs) > self.MAX_TOKEN:
            subtoken1_idxs = subtoken1_idxs[len(subtoken1_idxs)-self.MAX_TOKEN:]
            subtoken1_idxs[0] = self.cls_vid

        subtoken2 = self.tokenizer.tokenize(s2) + [self.sep_token]
        subtoken2_idxs = self.tokenizer.convert_tokens_to_ids(subtoken2)
        if len(subtoken2_idxs) > self.MAX_TOKEN:
            subtoken2_idxs = subtoken2_idxs[:self.MAX_TOKEN]
            subtoken2_idxs[-1] = self.sep_vid

        src_subtoken_idxs = subtoken1_idxs + subtoken2_idxs
        segments_ids = [0] * len(subtoken1_idxs) + [1] * len(subtoken2_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
        

    def preprocess(self, sentences, labels):
        assert len(sentences) == len(labels)
        output = []
        for idx in range(len(sentences)):
            output.append(self.preprocess_one(sentences[idx], labels[idx]))
        return output


class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, tokenizer, data, idx, batch_size, device):
        PAD_ID = tokenizer.pad_token_id
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        seg = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src != PAD_ID)
        
        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.model = get_model(args.model_name, args.num_layers)
        self.linear = nn.Linear(self.model.config.hidden_size, args.vocab_label_size)
        self.dropout = nn.Dropout(0.2)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=args.vocab_label_size, reduction='sum')

    def forward(self, src, seg, mask_src):
        with torch.no_grad():
            top_vec, _ = self.model(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        clss = top_vec[:,0,:]
        final_rep = self.dropout(clss)
        conclusion = self.linear(final_rep) #batch_size * vocab_label_size
        return conclusion #batch_size * vocab_label_size
    
    def get_loss(self, batch_data):
        src, seg, label, mask_src = batch_data
        output = self.forward(src, seg, mask_src)
        return self.loss(output.view(-1,self.args.vocab_label_size), label.view(-1))

    def predict(self, src, seg, mask_src, label):
        output = self.forward(src, seg, mask_src)
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        return prediction, label


def prediction(tokenizer, dataset, model, args):
    preds = []; golds = []
    model.eval()
    for j in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(tokenizer, dataset, j, args.batch_size, args.device).get()
        pred, gold = model.predict(src, seg, mask_src, label)
        preds += pred
        golds += gold
    golds = [g.cpu().tolist() for g in golds]
    return accuracy_score(golds, preds), preds
