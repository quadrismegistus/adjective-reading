from .adjective_reading import *



def parse_keyword_sents(keyword, path_corpus=PATH_CORPUS, must_contain=MUST_CONTAIN, limit=None,**kwargs):
    for (url,sent) in iter_keyword_sents(keyword=keyword, path_corpus=path_corpus, must_contain=must_contain, limit=limit, **kwargs):
        if keyword and keyword.lower() not in sent.lower():
            continue
        yield (url, sent, get_nlp_doc(sent,id=url))

def iter_parsed_sents(limit=None,**kwargs):
    import stanza
    avail = len(STASH_NLP)
    total = limit if limit and limit < avail else avail
    for i, (key, val) in enumerate(STASH_NLP.items(), 1):
        if i > total:
            break

        if len(key)==3 and key[0]:
            url, sentstr = key
            sentdoc = stanza.Document.from_serialized(val)
            yield url, sentstr, sentdoc



def get_parsed_sent_stats(sentdoc, keyword = None):
    tok_ld = sentdoc.to_dict()
    tok_id2obj = {}
    tok_id2d = {}
    tok_id2deprel = {}
    tok_id2head = {}
    tok_dd = {}

    for sent in sentdoc.sentences:
        for tok in sent.tokens:
            tok_d = tok.to_dict()[0]
            tok_id = tok_d['id']
            tok_id2d[tok_id] = tok_d
            tok_id2obj[tok_id] = tok

    out_ld = []
    for tok_id, tok_d in tok_id2d.items():
        tok_text = tok_d['text']
        if not keyword or tok_text.lower() == keyword.lower():
            tok_head_id = tok_d.get('head',0)
            tok_head_d = tok_id2d.get(tok_head_id,{})
            tok_id2head[tok_id] = tok_head_id
            ttext = tok_d.get("text","[text]")
            tlemma = tok_d.get("lemma",ttext)
            thead = tok_head_text = tok_head_d.get("text","[head]").lower()
            tdeprel = tok_id2deprel[tok_id] = tok_d.get('deprel',"[rel]")
            all_rels = [
                {
                    'id':w.id,
                    'deprel':w.deprel,
                    'head':w.head,
                }
                for w in sent.words
                if w.head == tok_id
            ]
            pprint(all_rels)
            
            token_ids_uppercase = [dx['id'] for dx in all_rels]
            sentstr2 = detokenize_and_uppercase(sentdoc, token_ids=token_ids_uppercase)

            for trel_d in all_rels:
                out_d = {
                    'source_id': tok_id,
                    'head_id': trel_d['id'],
                    'source': ttext,
                    'dep': trel_d['deprel'],
                    'head': trel_d['head'],
                    'sent': sentstr2
                }
            out_ld.append(out_d)
    return out_ld


def get_token_id2obj(sentdoc):
    tok_id2obj = {}
    for sent in sentdoc.sentences:
        for tok in sent.words:
            tok_id2obj[tok.id] = tok
    return tok_id2obj

# def get_token_span(sentdoc, tok_id_from, tok_id_to):
#     span = []
#     for sent in sentdoc.sentences:
#         for tok in sent.words:
#             if tok.id >= tok_id_from and tok.id <= tok_id_to:
#                 span.append(token)
#     return detokenize_and_uppercase(sentdoc, token_ids=span)

def get_token_info(sentdoc, tok_id):
    id2obj = get_token_id2obj(sentdoc)
    
    tok = id2obj[tok_id]
    tok_d = tok.to_dict()

    all_deps = []
    for id,tokx in id2obj.items():
        if tokx.head == tok_id:
            all_deps.append({'head_id':tokx.id,  'head_text':tokx.text, 'head_rel':tokx.deprel, 'head_type':'dep'})
        
    all_heads = []
    head_id = tok_d.get('head',0)
    if head_id and head_id in id2obj:
        head_tok = id2obj[head_id]
        head_tok_d = head_tok.to_dict()
        head_text = head_tok_d.get('text', '')
        head_deprel = head_tok_d.get('deprel', '')
        all_heads.append({'head_id':head_id, 'head_text':head_text, 'head_rel':tok_d.get('deprel', ''), 'head_type':'head'})
    
    all_rels = all_heads + all_deps

    out_ld = []
    for rel_d in all_rels:
        dephead = rel_d.get('head_type')
        tok_text = tok_d.get('text','')
        head_text = rel_d.get('head_text','')
        head_rel = rel_d.get('head_rel','')
        deprel = tok_d.get('deprel','')
        # if dephead == 'head':
            # rel_d['label'] = f"{tok_text} -- ({head_rel}) -- {head_text}"
        # else:
            # rel_d['label'] = f"{head_text} -- ({head_rel}) -- {tok_text}"
        rel_d['label'] = f"{head_rel}({head_text}, {tok_text})"
        out_d = {'id':tok_id,  **rel_d, 'text':tok_d.get('text', '')}
        out_ld.append(out_d)
    return [d for d in out_ld if d.get('head_rel')!='punct']
        
def get_tokens_info(sentdoc, keyword=None):
    out_ld = []
    for sent in sentdoc.sentences:
        for tok in sent.words:
            if keyword and tok.text.lower() != keyword.lower():
                continue
            out_ld.extend(get_token_info(sentdoc, tok.id))
    return out_ld

def iter_parsed_sents_stats(keyword, limit=None,**kwargs):
    for sent_id,(senturl,sentstr,sentdoc) in enumerate(parse_keyword_sents(keyword=keyword, limit=limit,**kwargs)):
        out_ld = get_tokens_info(sentdoc, keyword=keyword)
        for out_d in out_ld:
            sentstr2 = detokenize_and_uppercase(sentdoc, token_ids=[out_d['id'], out_d['head_id']])
            yield {'url':senturl, 'sent_id':sent_id, 'sent_str':sentstr, 'sent_repr':sentstr2, **out_d}

