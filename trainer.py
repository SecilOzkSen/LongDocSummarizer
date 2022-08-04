from dataset_loader import SummaryWithKeywordDataModule
from load_data import get_train_test_validation
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from model import LongDocumentSummarizerModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
print(torch.__version__)
print(torch.cuda.device_count())
print(torch.cuda.is_available())

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

MODEL_NAME_OR_PATH = 'allenai/longformer-base-4096'
N_EPOCHS = 1
BATCH_SIZE = 2

df_train, df_validation, df_test = get_train_test_validation()
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME_OR_PATH) #cls_token='[CLS]', sep_token='[SEP]')
longformer_model = LongformerModel.from_pretrained(MODEL_NAME_OR_PATH)
longformer_model.resize_token_embeddings(len(tokenizer))

data_module = SummaryWithKeywordDataModule(train_df=df_train,
                                           validation_df=df_validation,
                                           test_df=df_test,
                                           tokenizer=tokenizer,
                                           batch_size=BATCH_SIZE)

model = LongDocumentSummarizerModel(longformer_model, tokenizer, batch_size=BATCH_SIZE, cls_token_id=tokenizer.cls_token_id)

checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints",
                                      filename="summarizer-checkpoint",
                                      save_top_k=1,
                                      verbose=True,
                                      monitor="val_loss",
                                      mode="min")

logger = TensorBoardLogger("./logs",
                           name="longformer-summarizer",
                           )

trainer = pl.Trainer(logger=logger,
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=N_EPOCHS,
                     progress_bar_refresh_rate=30,
                     gpus=1,
                     accelerator='gpu'
                     )
if __name__ == "__main__":
    trainer.fit(model, data_module)
