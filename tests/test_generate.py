from wen.gpt import (
    TypinGPT,
    CacheItem,
    longest_substring_to_suffix_match,
)  # replace with your actual module and class
from wen.pinyin import PinYinTokenizer

engine = TypinGPT(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)


def test_pinyin_split():
    res = engine.split("nihao")


def test_encode_string_list():
    pinyin_tokenizer = PinYinTokenizer(pinyin2char=engine.pinyin2char)
    (
        token_list,
        pinyin_list,
        pinyin_id_list,
        input_ids,
    ) = pinyin_tokenizer.encode_string_list(["你好2008朋友"], engine.tokenizer)
    assert token_list == ["你", "好", "2008", "朋", "友"]
    assert pinyin_list == ["ni", "hao", None, "peng", "you"]
    assert pinyin_id_list[0] != 101  # no CLS special token
    assert input_ids[0] != 101  # no CLS special token, manually add if needed

    (
        token_list,
        pinyin_list,
        pinyin_id_list,
        input_ids,
    ) = pinyin_tokenizer.encode_string_list(["你好 friends "], engine.tokenizer)
    assert token_list == ["你", "好", "friends"]


def test_prepare_context():
    context_query, context_ids = engine.prepare_context("你好 friends ")
    assert context_query == ("你", "好", "friends")
    assert context_ids[0][0] == 101

    context_query, context_ids = engine.prepare_context("")
    assert context_query == ()
    assert context_ids.tolist() == [[101]]


def test_prepare_code_query():
    code_query = engine.prepare_code_query("nihao")
    assert code_query == ("ni", "hao")
    code_query = engine.prepare_code_query("")
    assert code_query == ()


def test_code_match_score():
    cache1 = CacheItem(context_key=(), code_key=("ni",))
    code_query = engine.prepare_code_query("niha")
    cache1.code_prefix_match(code_query)


def test_longest_matching_suffix():
    tuple_key = (1, 2, 3, 4, 5)
    tuple_query = (5, 6, 3, 4)
    assert longest_substring_to_suffix_match(tuple_query, tuple_key) == (2, 3)
    tuple_key = (4, 5)
    tuple_query = (5, 6, 3, 4)
    assert longest_substring_to_suffix_match(tuple_query, tuple_key) == (1, 0)


def test_match_key():
    assert engine.history == []
    # step 1
    cache1 = CacheItem(context_key=(), code_key=("ni",))
    context_query, context_ids = engine.prepare_context("")
    code_query = engine.prepare_code_query("niha")
    cache1.match_key(
        context_query,
        code_query,
    )

    # step 2
    cache2 = CacheItem(context_key=(), code_key=("ni", "ha"))
    context_query, context_ids = engine.prepare_context("")
    code_query = engine.prepare_code_query("nihao")

    # step 3
    cache3 = CacheItem(context_key=(), code_key=("ni", "hao"))
    context_query, context_ids = engine.prepare_context("")
    code_query = engine.prepare_code_query("nihaopen")

    # step 4
    cache4 = CacheItem(context_key=(), code_key=("ni", "hao", "pen"))
    context_query, context_ids = engine.prepare_context("")
    code_query = engine.prepare_code_query("nihaopeng")

    # step 5
    cache5 = CacheItem(context_key=(), code_key=("ni", "hao", "peng"))
    context_query, context_ids = engine.prepare_context("")
    code_query = engine.prepare_code_query("nihaopengyo")

    # step 6
    cache6 = CacheItem(context_key=(), code_key=("ni", "hao", "peng", "yo"))
    context_query, context_ids = engine.prepare_context("")
    code_query = engine.prepare_code_query("nihaopengyou")

    engine.history.extend([cache1, cache2, cache3, cache4, cache5, cache6])

    engine.clear_history()


def test_search_history_online():
    """
    test search history algoritm with online data
    """
    assert res[0] == "你好"
    assert (
        engine.last.pkv[0][0].size(-2) == engine.last.input_ids.size(-1) - 1
    ), "the length of past_key_values should be less than the length of input_ids"

    context = "你好"
    context_query, pinyin_query = engine.create_query(context)
    assert context_query == ("你", "好")
    background, context_ids, pkv, beam_history = engine.search_history(
        context_query + pinyin_query
    )

    assert len(pkv) == 12
    assert len(pkv[0]) == 2
    assert pkv[0][0].shape == pkv[0][1].shape
    assert list(pkv[0][0].shape) == [engine.num_beams, 12, 2, 64]
    assert background == (("你", "好"),) * engine.num_beams

    pass


def test_generate_on_context():
    # continue generate
    res = engine.generate("你好", "ma")
    assert res[0] == "吗"
    assert list(engine.last.pkv[0][0].shape) == [10, 12, 3, 64]
    assert engine.last.candidates_list[0] == ("你", "好", "吗")


def test_generate_on_partial_context():
    res = engine.generate("你好", "mapengyou")
    assert res[0] == "吗朋友"


def test_generate_on_fail_search():
    res = engine.generate("知道", "wozoule")
    assert res[0] == "我走了"
    assert len(engine.history) == 4
    res = engine.generate("我想认真", "henitaolunzhegewenti")
    assert res[0] == "和你讨论这个问题"
