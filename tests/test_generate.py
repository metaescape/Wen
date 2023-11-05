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

    # normal case
    engine.recent_key_values = None
    engine.generate("你好", "ma")
    assert engine.recent_key_values[0][0].size(2) == 3

    # continue generate
    engine.generate("", "nihao")
    engine.generate("你好", "ma")
    engine.generate("你好吗,", "zoule")
