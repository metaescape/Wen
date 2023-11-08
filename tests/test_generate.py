from wen.gpt import TypinGPT  # replace with your actual module and class

engine = TypinGPT(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)


def test_cls_past_key_values():
    engine.init_cls_kv()
    assert len(engine.cls_kv) == 12


def test_past_key_values_as_generate_inpu():
    # empty context
    res = engine.generate("", "nihao")
    assert res[0] == "你好"
    assert (
        engine.last.pkv[0][0].size(-2) == engine.last.input_ids.size(-1) - 1
    ), "the length of past_key_values should be less than the length of input_ids"

    background, context_ids, pkv, beam_history = engine.search_history("你好")
    assert len(pkv) == 12
    assert len(pkv[0]) == 2
    assert pkv[0][0].shape == pkv[0][1].shape
    assert list(pkv[0][0].shape) == [1, 12, 2, 64]

    # continue generate
    res = engine.generate("你好", "ma")
    assert res[0] == "吗"
    assert list(engine.last.pkv[0][0].shape) == [10, 12, 3, 64]
    assert engine.last.candidates_list[0] == ("你", "好", "吗")

    engine.generate("你好吗,", "zoule")
