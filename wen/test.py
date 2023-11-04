import re
from collections import defaultdict

PATTERN_RIME_DICT_ITEM = re.compile(
    r"^(?P<word>\w+)\t(?P<pinyin>[a-z ]+)(\t(?P<freq>[\d.]+)%?)?$"
)
# e.g. 土豆	tu dou	1777511


def load_rime_pinyin_dict(path):
    pinyin_dict = defaultdict(lambda: defaultdict(int))
    with open(path, mode="r") as f:
        for line in f:
            match = PATTERN_RIME_DICT_ITEM.match(line)
            if match:
                item = match.groupdict()
                pinyin, freq, word = item["pinyin"], item["freq"], item["word"]
                freq = int(freq) if freq else 1
                old_freq = pinyin_dict[pinyin][word]
                pinyin_dict[pinyin][word] = max(old_freq, freq)
    return pinyin_dict


PATTERN_RIME_USERDB_ITEM = re.compile(
    r"^(?P<pinyin>[a-z ]+)\t(?P<word>\w+)(\tc=.*t=(?P<freq>[\d.]+)%?)?$"
)
# an jian 	按键	c=28 d=1.48968 t=64151


def load_rime_userdb(path):
    pinyin_dict = defaultdict(lambda: defaultdict(int))
    with open(path, mode="r") as f:
        for line in f:
            match = PATTERN_RIME_USERDB_ITEM.match(line)
            if match:
                item = match.groupdict()
                pinyin, freq, word = item["pinyin"], item["freq"], item["word"]
                freq = int(freq) if freq else 1
                old_freq = pinyin_dict[pinyin][word]
                pinyin_dict[pinyin][word] = max(old_freq, freq)
    print(pinyin_dict)
    return pinyin_dict


load_rime_pinyin_dict("/home/pipz/myconf/rime/dicts/food.dict.yaml")
load_rime_userdb("/home/pipz/myconf/rime/extended.userdb.txt")
