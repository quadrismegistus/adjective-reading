
import os
import orjsonl
from tqdm import tqdm
import nltk
import re
from pprint import pprint
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import plotnine as p9
from tqdm import tqdm

PATH_CORPUS = '/Users/ryan/github/ordinary-style-philosophy/data/raw/LitStudiesJSTOR.jsonl'
MIN_WORDS = 3
EXCL_NUMBERS = True
EXCL_FIRST = True
EXCL_LAST = True
MUST_CONTAIN = ["reading"]
CONTEXT_SIZE = 100
CORPUS_NUM_SENTS = 71902

PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_DATA = os.path.join(PATH_REPO, 'data')

PATH_CORPUS_SENTS = os.path.join(PATH_DATA, 'corpus_sents.pkl.gz')
PATH_INSTANCES = os.path.join(PATH_DATA, 'corpus_instances.pkl.gz')
PATH_WORD2DATA = os.path.join(PATH_DATA, 'word2data.pkl.gz')

word2pos_fn = '/Users/ryan/Dropbox/Share/data/byu_word_data/worddb.byu.txt'
WORD2DATA = None
def get_word2data():
    global WORD2DATA
    if WORD2DATA is None:
        print(f'Loading word2data from {word2pos_fn}...')
        df = pd.read_csv(word2pos_fn, sep='\t').set_index('word')
        WORD2DATA = {word: dict(row) for word, row in df.iterrows()}
    return WORD2DATA


from hashstash import HashStash
STASH_NLP = HashStash('adjread_nlp')
# STASH_NLP.clear()
NLP = None

def get_nlp():
    global NLP
    if NLP is None:
        import stanza
        NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner,depparse,constituency', verbose=0)
    return NLP

def get_nlp_doc(txt,id=None):
    key = (id,txt)
    if key in STASH_NLP:
        import stanza
        return stanza.Document.from_serialized(STASH_NLP[key])
    nlp = get_nlp()
    doc = nlp(txt)
    STASH_NLP[key] = doc.to_serialized()
    return doc



def iter_keyword_sents(keyword, path_corpus=PATH_CORPUS, must_contain=MUST_CONTAIN, limit=None,**kwargs):
    keyword_l = keyword.lower()
    num = 0
    for text_i,d in enumerate(tqdm(orjsonl.stream(path_corpus), total=CORPUS_NUM_SENTS)):
        for page_i,page in enumerate(d['fullText']):
            page_proc = process_page(page, must_contain=keyword or must_contain, proc_sent=False, **kwargs)
            for sent_i,sent in enumerate(page_proc):
                if keyword_l in sent.lower():
                    yield {
                        'url': d['url'],
                        'text_id': text_i,
                        'page_id': page_i,
                        'sent_id': sent_i,
                        'sent': sent
                    }
                    num += 1
                    if limit and num >= limit:
                        return


def get_instances_data(force=False, **kwargs):
    if not force and os.path.exists(PATH_INSTANCES):
        return pd.read_pickle(PATH_INSTANCES)
    else:
        odf = pd.DataFrame(yield_sents_by_match(**kwargs))
        ofn_dir = os.path.dirname(PATH_INSTANCES)
        os.makedirs(ofn_dir, exist_ok=True)
        odf.to_pickle(PATH_INSTANCES)
        return odf










def yield_sents(path_corpus: str = PATH_CORPUS, **kwargs):
    for d in tqdm(orjsonl.stream(path_corpus), total=CORPUS_NUM_SENTS):
        for text_d in process_text(d['fullText'], **kwargs):
            yield {
                'url': d['url'],
                **text_d
            }

def process_text(pages: list[str], **kwargs):
    for page_num, page_sents in enumerate([process_page(page, **kwargs) for page in pages]):
        for sent_d in page_sents:
            yield {
                'page_num': page_num+1,
                **sent_d
            }

def process_page(page: str, min_words: int = 3, excl_numbers: bool = EXCL_NUMBERS, excl_first: bool = EXCL_FIRST, excl_last: bool = EXCL_LAST, must_contain: list[str] = MUST_CONTAIN, proc_sent: bool = True):
    sents = nltk.sent_tokenize(page)

    sentid2sent = {i: sent for i, sent in enumerate(sents)}

    filtered_sentid2sent = {**sentid2sent}

    # remove first and last sentence
    if excl_first:
        if 0 in filtered_sentid2sent:
            del filtered_sentid2sent[0]
    if excl_last:
        if len(sents) - 1 in filtered_sentid2sent:
            del filtered_sentid2sent[len(sents) - 1]
    if excl_numbers:
        filtered_sentid2sent = {k: v for k, v in filtered_sentid2sent.items() if not re.search(r'\d', v)}
    if min_words:
        # remove sents with fewer than n words
        filtered_sentid2sent = {k: v for k, v in filtered_sentid2sent.items() if len(v.split()) >= min_words}
    if must_contain:
        filtered_sentid2sent = {k: v for k, v in filtered_sentid2sent.items() if any(x.lower() in v.lower() for x in must_contain)}

    # convert to str    
    out_ld = [process_sent(sentid2sent, k, must_contain=must_contain) if proc_sent else sentid2sent[k] for k,v in sorted(filtered_sentid2sent.items())]
    return out_ld






def get_prev_sentences(sentences: list[str], target_index: int, context_size: int = CONTEXT_SIZE) -> list[str]:
    if target_index == 0 or context_size == 0:
        return []
    
    prev_sentences = []
    word_count = 0
    
    for i in range(target_index - 1, -1, -1):
        sent = sentences[i]
        words = sent.split()
        if word_count + len(words) <= context_size:
            prev_sentences.insert(0, sent)
            word_count += len(words)
        else:
            break
    return prev_sentences

def get_next_sentences(sentences: list[str], target_index: int, context_size: int=CONTEXT_SIZE) -> list[str]:
    if target_index == len(sentences) - 1 or context_size == 0:
        return []
    
    next_sentences = []
    word_count = 0
    
    for i in range(target_index + 1, len(sentences)):
        sent = sentences[i]
        words = sent.split()
        if word_count + len(words) <= context_size:
            next_sentences.append(sent)
            word_count += len(words)
        else:
            break
    return next_sentences

def get_next_context(sentences: list[str], target_index: int, context_size: int=CONTEXT_SIZE) -> str:
    next_sentences = get_next_sentences(sentences, target_index, context_size=context_size)
    return '\n'.join(next_sentences)

def get_prev_context(sentences: list[str], target_index: int, context_size: int=CONTEXT_SIZE) -> str:
    prev_sentences = get_prev_sentences(sentences, target_index, context_size=context_size)
    return '\n'.join(prev_sentences)


def process_sent(sentid2str: dict[int, str], sent_id: int, context_size: int = CONTEXT_SIZE, must_contain: list[str] = MUST_CONTAIN):
    sent_num = sent_id+1
    sentences_list = list(sentid2str.values())
    prev_context = get_prev_context(sentences_list, sent_id, context_size=context_size)
    next_context = get_next_context(sentences_list, sent_id, context_size=context_size)
    sent_text = sentid2str[sent_id]
    sent_text_refmt = sentid2str[sent_id]
    if must_contain:
        for x in must_contain:
            for y in {x, x.lower()}:
                if y in sent_text_refmt:
                    sent_text_refmt = sent_text_refmt.replace(y, f'__{y}__')
    return {
        'sent_num': sent_num,
        'sent': sent_text_refmt,
        'context0': prev_context,
        'context1': sent_text,
        'context2': next_context,
    }








def printm(s):
    from IPython.display import Markdown, display
    display(Markdown(s))

import string

def strip_punct(s):
    """Remove punctuation from beginning and end of string."""
    return s.strip(string.punctuation)    


def yield_sents_by_match(match_keywords=MUST_CONTAIN, **kwargs):
    text_iter = yield_sents(must_contain=match_keywords, **kwargs)
    for text in text_iter:
        for keyword in match_keywords:
            if keyword in text['sent']:
                for match_d in process_match(text, keyword):
                    yield {
                        **text,
                        **match_d
                    }


def process_match(text_d: dict, keyword: str):
    sent_str = text_d['context1']
    assert keyword in sent_str, f'keyword {keyword} not in {sent_str}'
    word2data = get_word2data()
    tokens = sent_str.split()
    for token_id, token in enumerate(tokens):
        if not keyword.lower() in token.lower():
            continue
        
        # get prev alpha token
        nexttok = ""
        prevtok = ""
        for tok in reversed(tokens[:token_id]):
            if any(x.isalpha() for x in tok):
                prevtok = tok
                break
        for tok in tokens[token_id+1:]:
            if any(x.isalpha() for x in tok):
                nexttok = tok
                break
        
        yield {
            "token_num": token_id,
            'token0': strip_punct(prevtok),
            'token1': strip_punct(keyword),
            'token2': strip_punct(nexttok),
            "word": keyword,
            **text_d,
            **(word2data[keyword] if keyword in word2data else {})
        }

def detokenize_and_uppercase(doc, token='reading', token_ids=[]):
    out = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            tok_id = tok.to_dict()[0]['id']
            text = tok.text
            if token_ids and tok_id in token_ids:
                text = text.upper()
            elif token and tok.text.lower() == token.lower():
                text = text.upper()
            
            out.append(text)
            out.append(tok.spaces_after)
    return ''.join(out)

# detokenize_and_uppercase(sentdoc, token_ids=[])