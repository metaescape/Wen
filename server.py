#!/usr/bin/env python
from pygls.capabilities import COMPLETION
from pygls.server import LanguageServer
from pygls.lsp import CompletionItem, CompletionList, CompletionOptions,\
    CompletionParams
from pygls.lsp.types import Range, TextEdit
import logging
import os
import sys
import re
from GodTian_Pinyin import GodTian_Pinyin

proj_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(proj_dir)
DEBUG = False

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

server = LanguageServer()

godtian = GodTian_Pinyin()

# @server.feature(COMPLETION, CompletionOptions(trigger_characters=[',']))
@server.feature(COMPLETION)
def completions(params: CompletionParams):
    """Returns completion items."""
    pos = params.position
    doc = server.workspace.get_document(params.text_document.uri)
    cur_word = doc.word_at_position(pos)
    if DEBUG == True:
        logger.debug(cur_word)

    if cur_word.isalpha(): 
        return trans(cur_word, pos)


def trans(prefix, pos):
    if prefix not in godtian.cache:
        hanzi, two_part = godtian.handle_current_input(prefix, 15, 15)
        godtian.cache[prefix] = hanzi
    else:
        hanzi = godtian.cache[prefix]

    items = []
    for ans in hanzi:
        cand = "".join(ans.path)
        text = TextEdit(range=Range(start=pos, end=pos), new_text=cand)
        item = CompletionItem(label=prefix)
        item.text_edit=text
        items.append(item)
    return CompletionList(is_incomplete=False, items=items)

server.start_io()
