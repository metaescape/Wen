from wen.gpt import IME
import torch
import pytest


engine = IME(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)
pinyin_tokenizer = engine.code_tokenizer


def test_tokenizer():
    
    (
        token_list,
        pinyin_list,
        pinyin_id_list,
        input_ids_list,
    ) = pinyin_tokenizer.encode_string_list(["你好2008朋友"], engine.tokenizer)
    assert token_list == ["你", "好", "2008", "朋", "友"]
    assert pinyin_list == ["ni", "hao", None, "peng", "you"]
    assert pinyin_id_list[0] != 101  # no CLS special token for pinyin ids
    
    assert input_ids_list[0] != 101  # no CLS special token, manually add if needed
    assert input_ids_list == [872, 1962, 8182, 3301, 1351]

    (
        token_list,
        pinyin_list,
        pinyin_id_list,
        input_ids_list,
    ) = pinyin_tokenizer.encode_string_list(["你好 friends "], engine.tokenizer)
    assert token_list == ["你", "好", "friends"]

    context_ids = engine.tokenizer.encode("你好2008朋友")
    assert context_ids == [101, 872, 1962, 8182, 3301, 1351, 102]

    context_ids = engine.tokenizer.encode("你好 friends ")
    assert context_ids == [101, 872, 1962, 11488, 102]
    

def test_model_generate_with_context():
    (
        token_list,
        pinyin_list,
        pinyin_id_list,
        input_ids_list,
    ) = pinyin_tokenizer.encode_string_list(["你好 friends "], engine.tokenizer)
    context_ids_list = [101] + input_ids_list

    current_code_tokens = engine.split("zaijian")
    assert current_code_tokens == ['zai', 'jian']

    context_ids = (
            torch.Tensor(context_ids_list).long().unsqueeze(0).to(engine.model.device)
        )
    context_len = context_ids.size(1)
    max_length = len(context_ids_list) + len(current_code_tokens)

    output_ids, last_key_values, beam_history = engine.model.generate(
            input_ids=context_ids,
            beam_scores=None,
            num_beams=16,
            num_return_sequences=10,
            logits_processor=engine.processors,
            max_length=max_length,
            constraint_pinyin_list=current_code_tokens,
            past_key_values=None,
        )
    
    res = []
    candidates_list = []
    target_ids = output_ids[:, context_len:]

    for i in range(target_ids.shape[0]):
        token_list = engine.tokenizer.convert_ids_to_tokens(target_ids[i])
        res.append("".join(token_list))
        candidates_list.append(tuple(token_list))
    
    assert res == ['再见', '在见', '在坚', '在建', '在减', '在健', '在艰', '在剑', '仔简', '再坚']
    assert candidates_list == [('再', '见'),
                               ('在', '见'),
                               ('在', '坚'),
                               ('在', '建'),
                               ('在', '减'),
                               ('在', '健'),
                               ('在', '艰'),
                               ('在', '剑'),
                               ('仔', '简'),
                               ('再', '坚')]

def test_IME_generate_with_context():
    candidates = engine.generate("你好 friends ", "zaijian")
    # default length is 9
    assert candidates[0] == '再见'
    assert len(candidates) == 9

    candidates = engine.generate("", "zai")
    # default length is 9
    assert candidates[0] == '在'
    assert len(candidates) == 9


    
    