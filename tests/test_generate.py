from wen.gpt import TypinGPT  # replace with your actual module and class

engine = TypinGPT(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)


def test_pinyin_split():
    res = engine.split("nihao")


def test_cls_past_key_values():
    engine.init_cls_kv()
    assert len(engine.cls_kv) == 12


def test_create_query():
    # Test case 1: without partial_context
    context = "你好"
    context_query, pinyin_query = engine.create_query(context)
    expected_result = tuple(engine.tokenizer.tokenize(context))
    assert context_query == expected_result

    # Test case 2: with partial_context
    partial_context = "ma"
    context_query, pinyin_query = engine.create_query(context, partial_context)
    expected_result = tuple(engine.split(partial_context))
    assert pinyin_query == expected_result


def test_search_history():
    # empty context
    res = engine.generate("", "nihao")
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
