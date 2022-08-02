from pytorch_lightning import LightningModule
import torch
from torch.optim import AdamW
from torch import nn
from rouge import Rouge
import math
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score
import spacy

rouge_v2 = Rouge()

class Classifier(nn.Module):
    def __init__(self, hidden_size, out_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h)
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=400):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]
        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class LongDocumentSummarizerModel(LightningModule):
    """
    The document summarization model which uses the pretrained Longfromer model.
    The input representations are computed with the methods in dataset_wrapper.py and passed to this model.
    """

    def __init__(self,
                 model,
                 tokenizer,
                 batch_size,
                 cls_token_id,
                 top_k: int = 5,
                 ):
        super().__init__()
        self.sentence_encoder_model = model
        self.tokenizer = tokenizer
        self.document_embedder = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.positional_encoding = PositionalEncoding(dropout=0.1, dim=768)
        self.classifier = Classifier(hidden_size=768, out_size=1)
        self.top_k = top_k
        self.gt_train_labels = pd.read_csv("training_labels.csv", delimiter='\t', converters={'labels': pd.eval})
        self.gt_validation_labels = pd.read_csv("validation_labels.csv", delimiter='\t', converters={'labels': pd.eval})
        self.gt_test_labels = pd.read_csv("test_labels.csv", delimiter='\t', converters={'labels': pd.eval})
        self.BCELoss = nn.BCELoss()
        self.spacy = spacy.load('en_core_web_sm')
        self.batch_size = batch_size
        self.cls_token_id = cls_token_id

    def get_global_attention_mask(self, input_ids, cls_token_indexes):

        attention_mask = torch.zeros(input_ids.shape, dtype=torch.long,
                                     device=input_ids.device)# initialize to local attention
        for i in range(self.batch_size):
            attention_mask[i, cls_token_indexes[i]] = 1
        return attention_mask

    def get_list_of_sentences(self, text):
        doc = self.spacy(text)
        lst = []
        for sent in doc.sents:
            lst.append(sent.text)
        return lst

    def pad_input(self, input, pad_dim=400):
        current_dim = input.size[1]
        return F.pad(input, pad=(0, 0, 0, pad_dim-current_dim, 0, 0), mode='constant', value=0)

    def get_cls_token_values_as_batch(self, last_hidden_state, cls_token_indexes):
        cls_token_values = []
        for i in range(self.batch_size):
            cls_token_index = cls_token_indexes[i]
            cls_token_index = torch.IntTensor(cls_token_index).to(self.device)
      #      print(cls_token_index)
            cls_token_value = torch.index_select(last_hidden_state[i], 0, cls_token_index)
            cls_token_values.append(cls_token_value)
        return np.array(cls_token_values)

    def forward(self, input_ids, labels, cls_token_indexes):
        global_attention_mask = self.get_global_attention_mask(input_ids, cls_token_indexes)
        output = self.sentence_encoder_model(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask
        )
        last_hidden_state = output.last_hidden_state
   #     cls_token_indexes = torch.IntTensor(cls_token_indexes).to(self.device)
   #     cls_token_values = torch.index_select(last_hidden_state, 1, cls_token_indexes)

        cls_token_values = self.get_cls_token_values_as_batch(last_hidden_state, cls_token_indexes)

        padded_output = self.pad_input(cls_token_values)

        positionally_encoded = self.positional_encoding(padded_output)

        document_embedder_output = self.document_embedder(positionally_encoded)

        results = self.classifier(document_embedder_output)

        return results

    def loss_calculation(self, predictions, ground_truth):
        predictions = torch.from_numpy(predictions)
        predictions = predictions.to(dtype=torch.double)
        ground_truth = torch.from_numpy(ground_truth)
        ground_truth = ground_truth.to(dtype=torch.double)
        loss = self.BCELoss(predictions, ground_truth)
        return loss

    def get_cls_token_indexes(self, input_ids, cls_token_id):
        batch_of_indexes = []
        for i in range(self.batch_size):
            np_input_ids = input_ids[i, :].flatten().cpu().detach().numpy()
            indexes = []
            for idx, token in enumerate(np_input_ids):
                if token == cls_token_id:
                    indexes.append(idx)
            batch_of_indexes.append(np.array(indexes))
        return np.array(batch_of_indexes)

    def calculate_F1(self, prediction, gt):
        return f1_score(gt, prediction)


    def produce_text_summary(self, predictions, text):
        doc = self.spacy(text)
        sentence_list = []
        for sent in doc.sents:
            sentence_list.append(sent.text)

        rounded = np.where(predictions > 0.6, 1., 0.)# threshold
        summary_sentences = []
        for idx, i in enumerate(rounded):
            if i == 1:
                summary_sentences.append(sentence_list[idx][0])
        summary = ' '.join(summary_sentences).strip()
        return summary, rounded



    def produce_summary_labels(self, results, text_sentence_length):
        results = results.flatten().cpu().detach().numpy()
        results = results[0:text_sentence_length]
        return results


    def produce_summary_input_ids(self, results, text_sentence_length, sentence_list, top_k=4):
        results = results.flatten().cpu().detach().numpy()
        results = results[0:text_sentence_length]
        sorted = np.flip(results.argsort(), axis=0)
        sorted = sorted[0:top_k]
        summary_sentences = [sentence_list[i][0] for i in sorted]
        produced_summary = ' '.join(summary_sentences)
        produced_summary_input_ids = self.tokenizer(
            produced_summary,
            max_length=512,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return produced_summary, produced_summary_input_ids

    def get_ground_truth_labels(self, document_id, source='training'):
        if source == "validation":
            df = self.gt_validation_labels
        elif source == "test":
            df = self.gt_test_labels
        else:
            df = self.gt_train_labels

        rows = df.loc[df['id'].isin(document_id)]
        gt = rows['labels'].tolist()
        return gt


    def training_step(self, batch, batch_idx):
        input_ids = batch["document_input_ids"]
        labels = batch["summary_input_ids"]
        text_sentence_length = batch["text_sentence_length"]
        document_id = batch["document_id"]
        cls_token_indexes = self.get_cls_token_indexes(input_ids, self.cls_token_id)
        ground_truth = self.get_ground_truth_labels(document_id, source="train")
        outputs = self.forward(input_ids=input_ids,
                               labels=labels,
                               cls_token_indexes=cls_token_indexes)
        predictions = self.produce_summary_labels(outputs, text_sentence_length)

        if len(predictions) != len(ground_truth):
            min_length = min(len(predictions), len(ground_truth))
            ground_truth = ground_truth[0:min_length]
            predictions = predictions[0:min_length]

        loss = self.loss_calculation(predictions, ground_truth)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
        return torch.autograd.Variable(loss, requires_grad=True)


    def validation_step(self, batch, batch_idx):
        text = batch['text']
        input_ids = batch["document_input_ids"]
        labels = batch["summary_input_ids"]
        text_sentence_length = batch["text_sentence_length"]
        document_id = batch["document_id"]
        gt_summary = batch["summary"]
        cls_token_indexes = self.get_cls_token_indexes(input_ids, self.cls_token_id)
        ground_truth = self.get_ground_truth_labels(document_id, source="validation")
        outputs = self.forward(input_ids=input_ids,
                               labels=labels,
                               cls_token_indexes=cls_token_indexes)

        predictions = self.produce_summary_labels(outputs, text_sentence_length)
        produced_summary, rounded_predictions = self.produce_text_summary(predictions, text)

        if len(predictions) != len(ground_truth):
            min_length = min(len(predictions), len(ground_truth))
            rounded_predictions = rounded_predictions[0:min_length]
            ground_truth = ground_truth[0:min_length]
            predictions = predictions[0:min_length]

        f1 = self.calculate_F1(rounded_predictions, ground_truth)

        loss = self.loss_calculation(predictions, ground_truth)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
        self.log("val_f1", f1 , prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)

        if produced_summary != '':
            scores = rouge_v2.get_scores(produced_summary, ' '.join(gt_summary).strip())
            scores = scores[0]
            rouge1 = scores['rouge-1']
            rouge2 = scores['rouge-2']
            rougeN = scores['rouge-l']

            self.log("rouge_1", rouge1['f'], prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
            self.log("rouge_2", rouge2['f'], prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
            self.log("rouge_n", rougeN['f'], prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["document_input_ids"]
        labels = batch["summary_input_ids"]
        text_sentence_length = batch["text_sentence_length"]
        document_id = batch["document_id"]
        cls_token_indexes = self.get_cls_token_indexes(input_ids, self.cls_token_id)
        outputs = self.forward(input_ids=input_ids,
                               labels=labels,
                               cls_token_indexes=cls_token_indexes)

        ground_truth = self.get_ground_truth_labels(document_id, source="test")
        predictions = self.produce_summary_labels(outputs, text_sentence_length)

        if len(predictions) != len(ground_truth):
            min_length = min(len(predictions), len(ground_truth))
            predictions = predictions[0:min_length]
            ground_truth = ground_truth[0:min_length]

        loss = self.loss_calculation(predictions, ground_truth)
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)
