import pandas as pda
import json
from pypinyin import lazy_pinyin, Style


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


def get_pinyin_to_char(tokenizer, pinyin2char_json):
    with open(pinyin2char_json) as f:
        pinyin2char = json.load(f)

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
