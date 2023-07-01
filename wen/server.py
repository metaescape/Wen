#!/home/pipz/miniconda3/envs/torch2/bin/python

from lsprotocol.types import (
    TEXT_DOCUMENT_COMPLETION,
    CompletionItem,
    CompletionList,
    CompletionParams,
    TEXT_DOCUMENT_COMPLETION,
    TEXT_DOCUMENT_DID_SAVE,
    TEXT_DOCUMENT_DID_OPEN,
    DidSaveTextDocumentParams,
    DidOpenTextDocumentParams,
    TextEdit,
    Range,
)


from pygls.server import LanguageServer

import logging
import os
import sys
import re
import json
from configuration import WenConfig

proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
sys.path.append(proj_dir)


DEBUG = True
use_godtian = True
use_pinyin_initial = False


def debug_logger():
    logger = logging.getLogger("pinyin")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler("/tmp/wen.log")
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger


logger = debug_logger()

try:
    import jieba
    import jieba.posseg as pseg

    jieba.enable_parallel()
    from pypinyin import lazy_pinyin
except:
    use_pinyin_initial = False

if use_godtian:
    im_dir = os.path.join(proj_dir, "./GodTian_Pinyin")
    logger.debug(proj_dir)
    os.chdir(im_dir)

    sys.path.append(im_dir)
    import GodTian_Pinyin as gp

    godtian = gp.GodTian_Pinyin()

else:
    godtian = False

logger.debug(f"this is {godtian}")

words = set(["中文语言服务", "wenls", "Metaescape"])

# TODO: use \' as pinyin separator
RE_START_WORD = re.compile("['A-Za-z_0-9\/]*$")

wen_cfg = WenConfig()


def generate_cands_from_godtian(prefix, context=None):
    """
    给定 prefix 和上下文，返回候选词列表。
    例如给定 prefix = nihao, 返回 ["你好","逆号"]
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


def items_from_triple_map(triples, pos, tail_space=False):
    """
    mainly for latex
    """
    items = []
    for prefix, show, insert in triples:
        if tail_space:
            insert = insert + " "
        text_edit = TextEdit(range=Range(start=pos, end=pos), new_text=show)
        item = CompletionItem(label=prefix)
        item.text_edit = text_edit
        item.insert_text = insert
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
        if not "\u4e00" <= _char <= "\u9fa5":
            return False
    return True


def both_noun(posseg_pair):
    a, b = posseg_pair
    return a.flag.startswith("n") and b.flag.startswith("n")


def cat_posseg_pair(posseg_pair):
    a, b = posseg_pair
    return "".join((a.word, b.word))


def _update_words(ls, params):
    ls.show_message_log("Updating words...")
    text_doc = ls.workspace.get_document(params.text_document.uri)
    source = text_doc.source
    for line in source.split("\n"):
        line_words = list(pseg.lcut(line))
        bigram = zip(line_words[:-1], line_words[1:])
        bigram = [bi for bi in bigram if both_noun(bi)]
        bi_words = [cat_posseg_pair(bi) for bi in bigram]
        words.update(bi_words)


def generate_pinyin_map_from_words(words):
    pinyin_map = {}
    for word in words:
        pinyin = lazy_pinyin(word)
        initials = "".join([ele[0] for ele in pinyin])
        pinyin_map[initials] = word
    return pinyin_map


latex_env_pair = {
    r"\(": r"\)",
    r"\[": r"\]",
    r"\begin": r"\end",
}


def latex_open_left(text):
    for left, right in latex_env_pair.items():
        left_idx = text.rfind(left)
        if left_idx >= 0:
            right_idx = text.rfind(right)
            if right_idx < left_idx:
                return left


def latex_open_right(text, left, right):
    right_idx = text.find(right)
    if right_idx >= 0:
        left_idx = text.find(left)
        if left_idx < 0 or right_idx < left_idx:
            return right


def in_latex_env(doc, pos):
    offset = doc.offset_at_position(pos)
    text_before = doc.source[:offset]
    text_after = doc.source[offset:]
    left = latex_open_left(text_before)
    if left:
        right = latex_env_pair[left]
        return latex_open_right(text_after, left, right) is not None
    return False


server = LanguageServer("wen-server", "v0.1")


# @server.feature(COMPLETION, CompletionOptions(trigger_characters=[',']))
@server.feature(TEXT_DOCUMENT_COMPLETION)
def completions(params: CompletionParams):
    """Returns completion items."""
    pos = params.position
    doc = server.workspace.get_document(params.text_document.uri)

    cur_word = doc.word_at_position(pos, RE_START_WORD)
    if DEBUG:
        logger.debug(cur_word)

    completion_list = CompletionList(is_incomplete=False, items=[])

    user_define_map = dict(
        [
            ("8alpha", "α"),
            ("8beta", "β"),
            ("8gamma", "γ"),
            ("8Delta", "Δ"),
            ("9/check", "- [ ] "),
        ]
    )

    # completion_list.add_items(items_from_map(user_define_map, pos))

    if in_latex_env(doc, pos) and cur_word.isalpha():
        completion_list.items.extend(
            items_from_triple_map(wen_cfg.latex_table, pos, True)
        )

    if use_pinyin_initial:
        pinyin_map = generate_pinyin_map_from_words(words)
        completion_list.items.extend(items_from_map(pinyin_map, pos))

    if godtian and (cur_word.isalpha() or cur_word.replace("'", "").isalpha()):
        pinyin = generate_cands_from_godtian(cur_word)
        logger.debug(pinyin)
        completion_list.items.extend(
            items_from_prefix_cands(cur_word, pinyin, pos)
        )

    return completion_list


@server.feature(TEXT_DOCUMENT_DID_SAVE)
def did_change(ls, params: DidSaveTextDocumentParams):
    """Text document did change notification."""
    _update_words(ls, params)


@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    ls.show_message("Text Document Did Open")
    _update_words(ls, params)


server.start_io()
