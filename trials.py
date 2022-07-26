import torch
from transformers import LongformerTokenizerFast, LongformerModel
from load_data import get_train_test_validation
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import numpy as np
if __name__ == "__main__":
    df_train, df_val, df_test = get_train_test_validation()
    col = df_train.iloc[0]
    splitter = SpacySentenceSplitter('en_core_web_sm')
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096") #cls_token='[CLS]', sep_token='[SEP]'
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

    cls_token_idx = tokenizer.cls_token_id
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    sentences = splitter.split_sentences(col["article"])
    str = ""
    for idx, sentence in enumerate(sentences):
        '''if idx == 0:
            formatted = f"{sentence} {sep_token} "
        elif idx == (len(sentences)-1):
            formatted = f"{cls_token} {sentence}"
        else:'''
        formatted = f"{cls_token} {sentence} {sep_token} "
        str = f"{str} {formatted}"

    input_ids = tokenizer(str.strip(), add_special_tokens=False, return_tensors='pt').input_ids

    indexes = []

    np_input_ids = input_ids.flatten().cpu().detach().numpy()
    for idx, token in enumerate(np_input_ids):
        if token == cls_token_idx:
            indexes.append(idx)
    print(indexes)

    attention_mask = torch.zeros(input_ids.shape, dtype=torch.long,
                                device=input_ids.device)  # initialize to local attention

    attention_mask[:, indexes] = 1

    output = model(input_ids, global_attention_mask=attention_mask, output_hidden_states=True)

    last_hidden_state = output.last_hidden_state

    cls_token_values = last_hidden_state.index_select(1, torch.IntTensor(indexes))
    print(cls_token_values)

