from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import get_model, bart_t5_forward
from torch.autograd import Variable
import torch
import torch.nn as nn


TOTAL_SENTENCE=2

def AvgPooling(data, denominator, dim=1):
    assert len(data.size()) == 3
    assert len(denominator.size()) == 1
    sum_data = torch.sum(data, dim)
    avg_data = torch.div(sum_data.transpose(1, 0), denominator).transpose(1, 0)
    return avg_data.contiguous()


class ModelData():
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.MAX_TOKEN_PREM = args.max_token_prem
        self.MAX_TOKEN_NEXT = args.max_token_next

    def preprocess_one(self, chat, resp, label):
        chat_subtokens = self.tokenizer.tokenize(chat)
        chat_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(chat_subtokens)
        if len(chat_subtoken_idxs) > self.MAX_TOKEN_PREM:
            chat_subtoken_idxs = chat_subtoken_idxs[len(chat_subtoken_idxs)-self.MAX_TOKEN_PREM:]

        resp_subtokens = self.tokenizer.tokenize(resp)
        resp_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(resp_subtokens)
        if len(resp_subtoken_idxs) > self.MAX_TOKEN_NEXT:
            resp_subtoken_idxs = resp_subtoken_idxs[:self.MAX_TOKEN_NEXT]
        
        return [chat_subtoken_idxs, resp_subtoken_idxs], label
    
    def preprocess(self, chats, resps, labels):
        assert len(chats) == len(resps) == len(labels)
        output = []
        for idx in range(len(chats)):
            output.append(self.preprocess_one(chats[idx], resps[idx], labels[idx]))
        return output


class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    def _get_max_token(self, batch):
        max_token_in_sentence = 0
        max_total_token = 0
        for tweets, labels in batch:
            total_token = 0
            for tweet in tweets:
                total_token += len(tweet)
                if max_token_in_sentence < len(tweet):
                    max_token_in_sentence = len(tweet)
            if max_total_token < total_token:
                max_total_token = total_token
        return max_token_in_sentence, max_total_token
    
    # do padding here
    def __init__(self, tokenizer, data, idx, batch_size, device):
        PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        cur_batch = data[idx:idx+batch_size]
        cur_batch_size = len(cur_batch)
        max_token_in_sentence, max_total_token = self._get_max_token(cur_batch)
        stack_index = Variable(torch.ones(cur_batch_size * TOTAL_SENTENCE * max_token_in_sentence)).type(torch.LongTensor) * max_total_token
        stack_denominator = Variable(torch.ones(cur_batch_size * TOTAL_SENTENCE)).type(torch.FloatTensor) * -1
        batch_tokens = []
        for idy in range(len(cur_batch)):
            cur_tokens = []
            offset = idy * TOTAL_SENTENCE * max_token_in_sentence
            offset_denom = idy * TOTAL_SENTENCE 
            tweet_offset = idy * (max_total_token + 1)
            for idt, tweet in enumerate(cur_batch[idy][0]):
                cur_tokens += tweet
                for idz, token in enumerate(tweet):
                    stack_index [offset + idz] = tweet_offset + idz
                stack_denominator[offset_denom + idt] = len(tweet)
                tweet_offset += len(tweet)
                offset += max_token_in_sentence
            batch_tokens.append(cur_tokens)

        src = torch.tensor(self._pad(batch_tokens, PAD_ID))
        label = torch.tensor([x[1] for x in cur_batch])
        mask_src = 0 + (src != PAD_ID)
        
        self.src = src.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)
        self.stack_index = stack_index.to(device)
        self.stack_denominator = stack_denominator.to(device)

    def get(self):
        return self.src, self.label, self.mask_src, self.stack_index, self.stack_denominator


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.model = get_model(args.model_name, args.num_layers)
        self.linear = nn.Linear(self.model.config.hidden_size*2, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction='none') 

    def forward(self, src, mask_src, stack_index, stack_denominator):
        batch_size = src.shape[0]
        with torch.no_grad():
            if self.args.model_type in ['t5', 'bart']:
                top_vec  = bart_t5_forward(self.model, src, mask_src, self.args.num_layers)
            elif self.args.model_type in ['electra']:
                top_vec = self.model(input_ids=src, attention_mask=mask_src)[0]
            else:
                top_vec, _ = self.model(input_ids=src, attention_mask=mask_src)
        batch_size, num_token, hidden = top_vec.shape
        bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        bucket = bucket.to(self.device)
        top_vec = torch.cat((top_vec, bucket), 1) # batch_size, action_num + 1, hidden_size
        top_vec = top_vec.view(batch_size * (num_token + 1), hidden)
        
        stack_state = torch.index_select(top_vec, 0, stack_index)
        stack_state = stack_state.view(batch_size * TOTAL_SENTENCE, -1, hidden)
        stack_state = AvgPooling(stack_state, stack_denominator)
        stack_state = stack_state.view(batch_size, TOTAL_SENTENCE, hidden)
        
        final_rep = self.dropout(stack_state) #batch_size * 2 * hidden_size
        final_rep = final_rep.view(batch_size, 1, hidden*2)
        conclusion = self.linear(final_rep).squeeze()
        return self.sigmoid(conclusion)
    
    def get_loss(self, batch_data):
        src, label, mask_src, stack_index, stack_denominator = batch_data
        output = self.forward(src, mask_src, stack_index, stack_denominator)
        return self.loss(output, label.float())
    
    def predict(self, src, mask_src, label, stack_index, stack_denominator):
        output = self.forward(src, mask_src, stack_index, stack_denominator)
        batch_size = output.shape[0]
        assert batch_size%2 == 0
        output = output.view(int(batch_size/2), 2)
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        answer = label.view(int(batch_size/2), 2)
        answer = torch.argmax(answer, dim=-1).data.cpu().numpy().tolist()
        return answer, prediction


def prediction(tokenizer, dataset, model, args):
    preds = []
    golds = []
    model.eval()
    assert len(dataset)%2==0
    assert args.batch_size%2==0
    for j in range(0, len(dataset), args.batch_size):
        src, label, mask_src, stack_index, stack_denominator = Batch(tokenizer, dataset, j, args.batch_size, args.device).get()
        answer, prediction = model.predict(src, mask_src, label, stack_index, stack_denominator)
        golds += answer
        preds += prediction
    return accuracy_score(golds, preds), preds
