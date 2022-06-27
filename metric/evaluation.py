from nltk.util import ngrams
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np


def compute_bleu(references, candidates):
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp = []
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
        else:
            ref_list.append([word_tokenize(references[i])])
    bleu1 = corpus_bleu(ref_list, dec_list,
                        weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 0, 0, 1))
    return {
        "bleu-1": bleu1,
        "bleu-2": bleu2,
        "bleu-3": bleu3,
        "bleu-4": bleu4,  # main result
    } 
    
def compute_meteor(references, candidates):
    score_list = []
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp =[]
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
            #ref_list = references[i]
        else:
            #ref_list = [references[i]]
            ref_list.append([word_tokenize(references[i])])
        score = meteor_score(ref_list[i], dec_list[i])
        score_list.append(score)
        
    return {
       "METEOR: ":  np.mean(score_list),
    }
    
    
def distinct_ngram(candidates, n=2):
    """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
    ngram_freqs = {}   # ngrams with frequencies
    ngram_len = 0  # total number of ngrams
    for candidate in candidates: 
        for ngram in ngrams(word_tokenize(candidate), n):
            ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
            ngram_len += 1
    # number of unique ngrams
    uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
    distinct_ngram = len(ngram_freqs) / ngram_len if ngram_len > 0 else 0
    print(f'Distinct {n}-grams:', round(distinct_ngram,4))
    return ngram_freqs, uniq_ngrams, ngram_len