# The code is borrowed and modified from BERTScore: https://github.com/Tiiiger/bert_score

import torch
from transformers import AutoModel
from transformers import __version__ as trans_version

model2hugmodel = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2',
    'electra': 'google/electra-base-discriminator',
    'gpt2': 'gpt2',
    'bart': 'facebook/bart-base',
    't5': 't5-small',
    'bert-large': 'bert-large-uncased',
    'bert-zh': 'bert-base-chinese',
    'bert-es': 'dccuchile/bert-base-spanish-wwm-uncased',
    'bert-de': 'bert-base-german-dbmdz-uncased'
}

model2layer = {
    'bert': 12,
    'roberta': 12,
    'albert': 12,
    'electra': 12,
    'gpt2': 12,
    'bart': 12,
    't5': 12,
    'bert-large': 24,
    'bert-zh': 12,
    'bert-es': 12,
    'bert-de': 12
}

def get_model(model_type, num_layers, all_layers=None):
    model = AutoModel.from_pretrained(model_type)
    model.eval()
    # drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"): #T5
                assert (
                    (len(model.encoder.block) +  len(model.encoder.block)) == 12
                ), f"We only handle T5-small in this experiment"
                # WARNING only handles T5-small
                if num_layers > 6:
                    model.decoder.block = torch.nn.ModuleList([layer for layer in model.decoder.block[:num_layers-6]])
                else:
                    model.encoder.block = torch.nn.ModuleList([layer for layer in model.encoder.block[:num_layers]])
                # model.encoder.block = torch.nn.ModuleList([layer for layer in model.encoder.block[:num_layers]])
            elif hasattr(model.encoder, "layers"): #BART and PEGASUS
                assert (
                    (len(model.encoder.layers) +  len(model.encoder.layers)) == 12
                ), f"We only handle BART-base in this experiment"
                # WARNING only handles BART-base
                if num_layers > 6:
                    model.decoder.layers =  model.decoder.layers[:num_layers-6]
                else:
                    model.encoder.layers = model.encoder.layers[:num_layers]
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        elif hasattr(model, "h"): #for GPT2
            assert (
                0 <= num_layers <= len(model.h)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.h)} for {model_type}"
            model.h = model.h[:num_layers]
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
        else:
            raise ValueError(f"Not supported model architecture: {model_type}")
    return model

def bart_t5_forward(model, x, attention_mask, num_layers):
    model.eval()
    if num_layers <=6:
        with torch.no_grad():
            out = model.encoder(x, attention_mask=attention_mask)
        return out[0]
    else:
        with torch.no_grad():
            out = model.forward(input_ids=x, attention_mask=attention_mask)
        return out[0]

