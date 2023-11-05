from wen.gpt import TypinGPT  # replace with your actual module and class

engine = TypinGPT(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)


def test_cls_past_key_values():
    engine.init_cls_kv()
    assert len(engine.cls_kv) == 12


def test_past_key_values_as_generate_inpu():
    engine.generate("你好", "ma")
    breakpoint()
