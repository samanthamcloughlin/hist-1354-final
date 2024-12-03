from datasets import load_dataset
from nltk.lm import Laplace
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize, sent_tokenize
import statistics
from unidecode import unidecode
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def train_model(corpus, ngram=7):
    tokenized_text = [list(word_tokenize(sent)) for sent in sent_tokenize(corpus)]

    lm = Laplace(ngram)
    train_data, vocab = padded_everygram_pipeline(ngram, tokenized_text)
    lm.fit(train_data, vocab)
    print("Trained on", len(corpus.split()), "words")
    return lm

def calc_perplexity(model, text, ngram=7):
    tokenized_text = [list(nltk.tokenize.word_tokenize(sent)) 
                    for sent in text]

    test_data, _ = padded_everygram_pipeline(ngram, tokenized_text)

    return statistics.mean([model.perplexity(test) for i, test in enumerate(test_data)])


def get_model_and_perplexity(pair, ngram=7, flip=False):
    data = load_dataset("jhu-clsp/kreyol-mt", pair)
    corpus_text = 'src_text' if flip else 'tgt_text'
    text_text = 'tgt_text' if flip else 'src_text'
    corpus = "\n".join([unidecode(x[corpus_text]) for x in data['train']['translation']])
    text = [unidecode(x[text_text]) for x in data['test']['translation']]

    model = train_model(corpus)
    perplexity = calc_perplexity(model, text, ngram)

    return model, perplexity

def calc_pld(pair):
    _, p_one = get_model_and_perplexity(pair)
    _, p_two = get_model_and_perplexity(pair,flip=True)
    ld = (p_one + p_two)/2
    return ld

def calc_pretrained_perplexity(texts, model, tokenizer):
    total_words = len(texts)
    perplexity = 0
    i = 0
    for text in texts:
        i+=1
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        if torch.isnan(loss):
            total_words -= 1
        else:
            perplexity += torch.exp(loss).item()

    return perplexity/total_words

if __name__ == "__main__":
    for p in ["djk-eng", "hat-eng", "hat-fra", "hat-ara", "hat-aze", 
              "hat-deu", "hat-nep", "hat-zho", "mfe-eng", "mfe-fra"]:
        print (p, calc_pld(p))

    for p in ['acf-eng', 'crs-eng', 'djk-eng', 'gcf-eng', 'hat-eng', 'kea-eng',
              'lou-eng', 'mfe-eng', 'pap-eng', 'pcm-eng', 'sag-eng', 'tpi-eng', 'trf-eng']:
        data = load_dataset("jhu-clsp/kreyol-mt", p)
        text = [unidecode(x['src_text']) for x in data['train']['translation']][:1000]
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        px = calc_pretrained_perplexity(text, model, tokenizer)
        print(p, px)