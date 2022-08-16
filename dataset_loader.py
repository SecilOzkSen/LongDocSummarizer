import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
import spacy
from datasets import Dataset
from transformers import LongformerTokenizer


class CNNDailyMailDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: LongformerTokenizer,
                 input_token_limit: int = 4000,
                 padding_limit: int = 400
                 ):
        self.dataset = dataframe
        self.tokenizer = tokenizer
        self.input_token_limit = input_token_limit
        self.spacy_model = spacy.load("en_core_web_sm")
        self.padding_limit = padding_limit

    def __len__(self):
        return len(self.dataset)

    def prepare_document(self, doc):
        if isinstance(doc, list):
            doc = ' '.join(doc)
        sentences = self.spacy_model(doc)
        new_doc = ""
        sentence_length = 0
        sentence_list = []
        for sentence in sentences.sents:
            if sentence_length >= self.padding_limit:
                break
            sentence_list.append(sentence.text)
            new_sentence = f"<s> {sentence.text} </s> "
            new_doc += new_sentence
            sentence_length += 1

        for i in range(0, self.padding_limit-sentence_length):
            sentence_list.append(' ')

        return new_doc.strip(), sentence_length, sentence_list

    def __getitem__(self, index:int):
        data_row = self.dataset.iloc[index]
        text = data_row["article"]
        summary = data_row["highlights"]
        id = data_row["id"]

        document, sentence_length, sentence_list = self.prepare_document(text)
        document_input_ids = self.tokenizer(
            document,
            max_length=self.input_token_limit,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        summary_input_ids = self.tokenizer(
            summary,
            max_length=512,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return dict(
            text=text,
            summary=summary,
            text_sentence_length=sentence_length,
            summary_input_ids=summary_input_ids.flatten(),
            document_input_ids=document_input_ids.flatten(),
            document_id=id
        )


class SummaryWithKeywordDataModule(LightningDataModule):
    def __init__(self,
                 train_df : pd.DataFrame,
                 validation_df : pd.DataFrame,
                 test_df : pd.DataFrame,
                 tokenizer: LongformerTokenizer,
                 batch_size: int = 64,
                 text_max_token_limit: int = 8192,
                 ):
        super().__init__()
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.text_max_token_limit = text_max_token_limit

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CNNDailyMailDataset(
            dataframe=self.train_df,
            tokenizer=self.tokenizer,
            input_token_limit=self.text_max_token_limit,
        )

        self.validation_dataset = CNNDailyMailDataset(
            dataframe=self.validation_df,
            tokenizer=self.tokenizer,
            input_token_limit=self.text_max_token_limit,
        )

        self.test_dataset = CNNDailyMailDataset(
            dataframe=self.test_df,
            tokenizer=self.tokenizer,
            input_token_limit=self.text_max_token_limit,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=20
                          )

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=20
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=20
                          )










