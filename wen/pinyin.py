from typing import List, Optional, Union

import pandas as pda
from pypinyin import Style, lazy_pinyin
from pypinyin.constants import RE_HANS
from pypinyin.converter import UltimateConverter
from pypinyin.core import Pinyin
from transformers import BertTokenizer


def get_pinyin(x):
    return lazy_pinyin(x, style=Style.TONE3, neutral_tone_with_five=True)


def get_pinyin_with_mode(words, abbr_mode):
    pinyin = get_pinyin(words)
    if abbr_mode == "xone":
        pinyin = [p[:-1] if i == 0 else p[0] for i, p in enumerate(pinyin)]
    elif abbr_mode == "full":
        pinyin = [p[0] for p in pinyin]
    elif abbr_mode == "none":
        pinyin = [p[:-1] for p in pinyin]  # remove tone
    else:
        raise ValueError("No such abbr mode!")
    return pinyin


def get_pinyin_to_char(tokenizer, pinyin2char: dict):
    df_vocab = pda.DataFrame(
        tokenizer.convert_ids_to_tokens(range(len(tokenizer.get_vocab()))),
        columns=["char"],
    )

    pinyin2chars = {}
    for p, wl in pinyin2char.items():
        df_tmp = df_vocab[df_vocab.char.isin(wl)]
        df_tmp = df_tmp.assign(idx=range(len(df_tmp)))
        pinyin2chars[p] = df_tmp

    return pinyin2chars


class PinYinTokenizer:
    def __init__(self, **kwargs):
        pinyin2char = {p: [] for p in "abcdefghijklmnopqrstuvwxyz"}
        self.first_letter = True
        if kwargs["pinyin2char"]:
            pinyin2char = kwargs["pinyin2char"]
            self.first_letter = False
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_to_id["[PAD]"] = 0
        self.id_to_token[0] = "[PAD]"

        for i, p in enumerate(pinyin2char.keys(), start=1):
            self.token_to_id[p] = i
            self.id_to_token[i] = p
        self.special_tokens_map = {
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
        }
        self._pinyin = Pinyin(
            UltimateConverter(
                v_to_u=False,
                neutral_tone_with_five=False,
                tone_sandhi=False,
            )
        )
        self.pinyin_args = {
            "style": Style.NORMAL,
            "errors": "default",
            "strict": True,
            "heteronym": False,
        }

    def _tokenize(self, text):
        raise Exception("currently not supported")

    def word_to_pinyin(self, word: str) -> List[str]:
        if self.first_letter:
            return lazy_pinyin(word, style=Style.FIRST_LETTER)
        return lazy_pinyin(word)

    def encode_string_list(
        self,
        hans: List[str],
        decoder_tokenizer: BertTokenizer,
    ) -> tuple[list[str], list[str], list[int], list[int]]:
        """convert hans to pinyin and then to pinyin_ids and input_ids
        return:
        token_list: list of tokenized words
        pinyin_list: list of pinyin
        pinyin_id_list: list of pinyin ids
        input_ids: list of input ids
        """

        token_list = []
        pinyin_list = []
        pinyin_id_list = []
        input_ids = []
        for word in hans:
            if not all_chinese_chars(word):
                for word_seg in self._pinyin.seg(word):
                    word_pinyin = self.word_to_pinyin(word_seg)
                    if word_pinyin[0] == word_seg:
                        # non chinese char, maybe split to subwords by bert
                        ids = decoder_tokenizer.encode(
                            word_seg, add_special_tokens=False
                        )
                        input_ids.extend(ids)
                        pinyin_list.extend([None] * len(ids))

                        token_list.extend(decoder_tokenizer.tokenize(word_seg))
                        pinyin_id_list.extend(self.encode([None] * len(ids)))
                    else:
                        word_pinyin = self.word_to_pinyin(word_seg)
                        token_list.extend(word_seg)
                        pinyin_list.extend(word_pinyin)
                        pinyin_id_list.extend(self.encode(word_pinyin))
                        input_ids.extend(
                            decoder_tokenizer.encode(
                                word_seg, add_special_tokens=False
                            )
                        )
            else:  # pure chinese word
                word_pinyin = self.word_to_pinyin(word)
                token_list.extend(word)
                pinyin_list.extend(word_pinyin)
                pinyin_id_list.extend(self.encode(word_pinyin))
                input_ids.extend(
                    decoder_tokenizer.encode(word, add_special_tokens=False)
                )

        return token_list, pinyin_list, pinyin_id_list, input_ids

    def encode(self, lst_or_pinyin: Optional[Union[list, str]]):
        if not isinstance(lst_or_pinyin, list):
            return self._convert_token_to_id(lst_or_pinyin)
        return [self._convert_token_to_id(x) for x in lst_or_pinyin]

    def _convert_token_to_id(self, token):
        return self.token_to_id.get(token, 600)  # 600 for non chinese char

    def is_vaild_id(self, token_id):
        return token_id in self.id_to_token

    def _convert_id_to_token(self, id):
        return self.id_to_token[id]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def all_chinese_chars(sentence):
    return RE_HANS.match(sentence) is not None


def is_chinese_char(char):
    """ 判断是否为中文字符且排除中文标点 """
    if '\u4e00' <= char <= '\u9fff':
        return True
    if '\u3400' <= char <= '\u4dbf':
        return True
    if '\u20000' <= char <= '\u2a6df':
        return True
    if '\u2a700' <= char <= '\u2b73f':
        return True
    if '\u2b740' <= char <= '\u2b81f':
        return True
    if '\u2b820' <= char <= '\u2ceaf':
        return True
    if '\u2ceb0' <= char <= '\u2ebef':
        return True
    if '\u30000' <= char <= '\u3134f':
        return True
    return False