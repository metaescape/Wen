#!/usr/bin/python3

import asyncio
import logging
import os
import re
import sys
from typing import Any, List, Optional, Union
import cattrs
import time

from lsprotocol.types import (
    INITIALIZE,
    TEXT_DOCUMENT_COMPLETION,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionOptions,
    CompletionParams,
    ConfigurationParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    InitializeParams,
    InitializeResult,
    MessageType,
    Range,
    ServerCapabilities,
    TextDocumentSyncKind,
    TextEdit,
)
from pygls.protocol import LanguageServerProtocol, lsp_method
from pygls.server import LanguageServer

__version__ = "0.0.1"

from wen.configuration import WenConfig
from wen.gpt import TypinGPT, IME
from wen.latex import in_latex_env

CFG = WenConfig()


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
logger.debug("start logging")

if CFG.use_pinyin_initial:
    import jieba
    import jieba.posseg as pseg

    jieba.enable_parallel()
    from pypinyin import lazy_pinyin


if CFG.completion == "demo":
    im_dir = os.path.join(proj_dir, "./GodTian_Pinyin")
    logger.debug(os.path)
    os.chdir(im_dir)

    sys.path.append(im_dir)
    import GodTian_Pinyin as gp

    godtian = gp.GodTian_Pinyin()

elif CFG.completion == "gpt2":
    # typinG = TypinGPT(model_name_or_path=CFG.model_path)
    typinG = IME(model_name_or_path=CFG.model_path)

words = set(["中文语言服务", "wenls", "Metaescape"])

# TODO: use \' as pinyin separator
RE_START_WORD = re.compile("['A-Za-z_0-9\/]*$")


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


def convert_suggestion_to_completion_item(pinyin, cand, pos):
    label = cand  # Assuming cand is already a string
    kind = CompletionItemKind.Text
    filter_text = pinyin  # Assuming cand is a suitable filter text
    text_edit = TextEdit(
        range=Range(
            start=pos, end=pos  # Assuming pos is already a position object
        ),
        new_text=cand,
    )

    completion_item = CompletionItem(
        label=label, kind=kind, filter_text=filter_text, text_edit=text_edit
    )

    return completion_item


def items_from_prefix_cands(prefix, cands, pos):
    items = [
        convert_suggestion_to_completion_item(prefix, cand, pos)
        for cand in cands
    ]
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


class WenLanguageServerProtocol(LanguageServerProtocol):
    """Override some built-in functions."""

    _server: "WenLanguageServer"

    @lsp_method(INITIALIZE)
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
        """Override built-in initialization.

        Here, we can conditionally register functions to features based
        on client capabilities and initializationOptions.
        """
        server = self._server
        try:
            server.initialization_options = (
                {}
                if params.initialization_options is None
                else params.initialization_options
            )

        except cattrs.BaseValidationError as error:
            msg = (
                "Invalid InitializationOptions, using defaults:"
                f" {cattrs.transform_error(error)}"
            )
            server.show_message(msg, msg_type=MessageType.Error)
            server.show_message_log(msg, msg_type=MessageType.Error)
        #  所有的初始化选项都在这里：
        logger.debug(params.initialization_options)

        initialize_result: InitializeResult = super().lsp_initialize(params)
        return initialize_result


class WenLanguageServer(LanguageServer):
    """Wen language server."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


wenls = WenLanguageServer(
    name="wen-server",
    version=__version__,
    protocol_cls=WenLanguageServerProtocol,
)

CFG.debug = True


@wenls.feature(TEXT_DOCUMENT_COMPLETION)
async def completions(params: CompletionParams):
    """Returns completion items."""
    pos = params.position
    doc = wenls.workspace.get_document(params.text_document.uri)

    line_context = doc.word_at_position(pos, re.compile(".*"))
    cur_word = doc.word_at_position(pos, RE_START_WORD)
    line_context = line_context[: -len(cur_word)]
    if CFG.debug:
        logger.debug("cur_word: %s", cur_word)

    completion_list = CompletionList(is_incomplete=True, items=[])

    # completion_list.add_items(items_from_map(user_define_map, pos))

    if in_latex_env(doc, pos) and cur_word.isalpha():
        completion_list.items.extend(
            items_from_triple_map(CFG.latex_table, pos, True)
        )

    if CFG.use_pinyin_initial and cur_word.isalpha():
        pinyin_map = generate_pinyin_map_from_words(words)
        completion_list.items.extend(items_from_map(pinyin_map, pos))

    if CFG.completion == "demo" and (
        cur_word.isalpha() or cur_word.replace("'", "").isalpha()
    ):
        cands = generate_cands_from_godtian(cur_word)

        completion_list.items.extend(
            items_from_prefix_cands(cur_word, cands, pos)
        )
    elif CFG.completion == "gpt2" and (
        cur_word.isalpha() or cur_word.replace("'", "").isalpha()
    ):
        completion_list.items.extend(
            items_from_prefix_cands(cur_word, [cur_word], pos)
        )
        loop = asyncio.get_event_loop()
        loop.create_task(
            generate_and_update_completions(
                line_context, cur_word, completion_list, pos
            )
        )
    

    return completion_list

import random
async def generate_and_update_completions(
    context, cur_word, completion_list, pos
):
    try:
        # 在后台执行 typinG.generate 函数
        # logger.debug("context: %s", context)
        cands = typinG.generate(context, cur_word, logger)
        # cands = [random.choice(["test", "test2", "test3", "test4"])]
        # await asyncio.sleep(0.2)
        # time.sleep(0.4)
        logger.debug(cands[0])
        # 更新补全列表
        completion_list.items.clear()
        completion_list.items.extend(
            items_from_prefix_cands(cur_word, cands, pos)
        )
    except:
        logger.debug("generate: %s", cands[0])


@wenls.feature(TEXT_DOCUMENT_DID_SAVE)
def did_change(ls, params: DidSaveTextDocumentParams):
    """Text document did change notification."""
    # _update_words(ls, params)
    pass


@wenls.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    ls.show_message("Text Document Did Open")
    _update_words(ls, params)


if __name__ == "__main__":
    wenls.start_io()
