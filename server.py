#!/usr/bin/env python
from pygls.capabilities import COMPLETION
from pygls.lsp.methods import (COMPLETION, TEXT_DOCUMENT_DID_CHANGE,
                               TEXT_DOCUMENT_DID_SAVE,
                               TEXT_DOCUMENT_DID_CLOSE, TEXT_DOCUMENT_DID_OPEN, 
                               TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL)
from pygls.lsp.types import (CompletionItem, CompletionList, CompletionOptions,
                             CompletionParams, ConfigurationItem, TextEdit,
                             ConfigurationParams, Diagnostic, 
                             DidChangeTextDocumentParams,
                             DidSaveTextDocumentParams,
                             DidCloseTextDocumentParams,
                             DidOpenTextDocumentParams, MessageType, Position,
                             Range, Registration, RegistrationParams)
from pygls.server import LanguageServer
from pygls.lsp import CompletionItem, CompletionList, CompletionOptions,\
    CompletionParams
import logging
import os
import sys
import re
proj_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(proj_dir) 

DEBUG = True
use_godtian = True
use_pinyin_initial = True

try:
    import jieba
    import jieba.posseg as pseg
    jieba.enable_parallel()
except:
    use_pinyin_initial = False

    

if use_godtian:
    im_dir = os.path.join(proj_dir, "./completion")
    os.chdir(im_dir)
    sys.path.append(im_dir) 
    from completion import GodTian_Pinyin as gp
    godtian = gp.GodTian_Pinyin()
else:
    godtian = False

def debug_logger():
    logger = logging.getLogger("pinyin")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('/tmp/wen.log')
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger

logger = debug_logger()

words = set(["??????????????????", "wenls", "Metaescape"])

# TODO: use \' as pinyin separator
RE_START_WORD = re.compile('[\'A-Za-z_0-9\/]*$')

def generate_cands_from_godtian(prefix, context=None):
    """
    ?????? prefix ???????????????????????????????????????
    ???????????? prefix = nihao, ?????? ["??????","??????"]
    """
    if prefix not in godtian.cache:
        hanzi, two_part = godtian.handle_current_input(prefix, 15, 15)
        godtian.cache[prefix] = hanzi
    else:
        hanzi = godtian.cache[prefix]
    return ["".join(ans.path) for ans in hanzi]


def items_from_map(symbol_map, pos):
    items = []
    for prefix, cand in symbol_map.items():
        text_edit = TextEdit(range=Range(start=pos, end=pos), new_text=cand)
        item = CompletionItem(label=prefix)
        item.text_edit = text_edit
        items.append(item)
    return items


def items_from_prefix_cands(prefix, cands, pos):
    items = []
    for cand in cands:
        text_edit = TextEdit(range=Range(start=pos, end=pos), new_text=cand)
        item = CompletionItem(label=prefix)
        item.text_edit = text_edit
        items.append(item)
    return items
        

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def both_noun(posseg_pair):
    a, b  = posseg_pair
    return a.flag.startswith("n") and b.flag.startswith("n")

def cat_posseg_pair(posseg_pair):
    a, b  = posseg_pair
    return "".join((a.word, b.word))
    

def _update_words(ls, params):
    ls.show_message_log('Updating words...')
    text_doc = ls.workspace.get_document(params.text_document.uri)
    source = text_doc.source
    for line in source.split("\n"):
        line_words = list(pseg.lcut(line))
        bigram = zip(line_words[:-1], line_words[1:])
        bigram = [bi for bi in bigram if both_noun(bi)]
        bi_words = [cat_posseg_pair(bi) for bi in bigram]
        words.update(bi_words)

from pypinyin import lazy_pinyin

def generate_pinyin_map_from_words(words):
    pinyin_map = {}
    for word in words:
        pinyin = lazy_pinyin(word)
        initials = "".join([ele[0] for ele in pinyin])
        pinyin_map[initials] = word
    return pinyin_map

server = LanguageServer()

# @server.feature(COMPLETION, CompletionOptions(trigger_characters=[',']))
@server.feature(COMPLETION)
def completions(params: CompletionParams):
    """Returns completion items."""
    pos = params.position
    doc = server.workspace.get_document(params.text_document.uri)

    cur_word = doc.word_at_position(pos, RE_START_WORD)
    logger.debug(doc)
    if DEBUG == True:
        logger.debug(cur_word)

    completion_list = CompletionList(is_incomplete=False, items=[])

    user_define_map = dict([
        ("8alpha",   "??"),
        ("8beta",   "??"),
        ("8gamma",   "??"),
        ("8Delta",  "??"),
        ("9/check",  "- [ ] ")
    ])

    completion_list.add_items(items_from_map(user_define_map, pos))

    if use_pinyin_initial:
        pinyin_map = generate_pinyin_map_from_words(words)
        completion_list.add_items(items_from_map(pinyin_map, pos))

    if godtian and (cur_word.isalpha() or cur_word.replace("'","").isalpha()): 
        pinyin = generate_cands_from_godtian(cur_word)
        completion_list.add_items(items_from_prefix_cands(cur_word, pinyin, pos))
    
    return completion_list

@server.feature(TEXT_DOCUMENT_DID_SAVE)
def did_change(ls, params: DidSaveTextDocumentParams):
    """Text document did change notification."""
    _update_words(ls, params)


@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    ls.show_message('Text Document Did Open')
    _update_words(ls, params)

server.start_io()
