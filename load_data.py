import numpy as np
from datasets import load_dataset
#from keybert import KeyBERT
#keybert = KeyBERT()
#from sentence_transformers import SentenceTransformer, util
import spacy

#sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
'''
nlp = spacy.load('en_core_web_sm')

def compare_sentences_summaries(text, summary, threshold = 0.65):
    text_doc = nlp(text)
    summary_doc = nlp(summary)

    text_sentences = []
    summary_sentences = []

    for sent in text_doc.sents:
        text_sentences.append(sent.text)

    for sent in summary_doc.sents:
        summary_sentences.append(sent.text)

    encoded_text = sentence_transformer_model.encode(text_sentences, convert_to_tensor=True)
    encoded_summary = sentence_transformer_model.encode(summary_sentences, convert_to_tensor=True)

    cosine_scores = util.cos_sim(encoded_text, encoded_summary).numpy()
    labels = np.zeros(shape=len(text_sentences), dtype=int)
    max_cosine_scores = [max(scores) for scores in cosine_scores]
    for idx, row in enumerate(max_cosine_scores):
        if row >= threshold:
            labels[idx] = 1

    if sum(labels.tolist()) == 0:
        idx = np.argmax(max_cosine_scores)
        labels[idx] = 1
    return labels.tolist()
    '''

def clean_dataset(dataset):
    def clean_article(txt: str):
        text = txt.split("--")
        if len(text) >= 2:
            merged = ""
            for chunk in text[1:]:
                merged = merged + " " + chunk
            return merged.strip()
        if len(text) == 1:
            text = ' '.join(text)
            return text.strip()
        return text
    dataset["article"] = dataset["article"].apply(lambda x: clean_article(x))
    return dataset


'''def extract_keywords(dataset):
    def extract_keyword(doc):
        keywords = keybert.extract_keywords(doc)
        keywords_list = []
        for keyword in keywords:
            keywords_list.append(keyword[0])
        return keywords_list
    dataset["keyword"] = dataset["article"].apply(lambda x: extract_keyword(x))
    return dataset'''

def get_train_test_validation(extract_keywords = False):
    training_data = load_dataset("cnn_dailymail", '3.0.0', split="train")
    validation_data = load_dataset("cnn_dailymail", '3.0.0', split="validation")
    test_data = load_dataset("cnn_dailymail", '3.0.0', split="test")
    training_data = clean_dataset(training_data.to_pandas())
    validation_data = clean_dataset(validation_data.to_pandas())
    test_data = clean_dataset(test_data.to_pandas())
    if extract_keywords:
        training_data = extract_keywords(training_data)
        validation_data = extract_keywords(validation_data)
        test_data = extract_keywords(test_data)
    return training_data, validation_data, test_data

'''if __name__ == '__main__':
    _, validation_data, test_data = get_train_test_validation()
    test_data["labels"] = test_data.apply(lambda x : compare_sentences_summaries(x.article, x.highlights), axis=1)
    test_data = test_data.drop(['article', 'highlights'], axis=1)
    test_data.to_csv("test_labels.csv",  sep='\t', encoding='utf-8')'''
