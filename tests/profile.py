
from wen.gpt import IME
from transformers import logging
logging.set_verbosity_error()
import argparse

import random
from pypinyin import lazy_pinyin
from wen.pinyin import is_chinese_char
import time



ime = IME(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)

print("Model loaded")


demo = """
工业互联网将整合两大革命性转变之优势：
其一是工业革命，伴随着工业革命，出现了无数台机器、设备、机组和工作站；
其二则是更为强大的网络革命，在其影响之下，计算、信息与通讯系统应运而生并不断发展。
伴随着这样的发展，三种元素逐渐融合，充分体现出工业互联网之精髓。
将这些元素融合起来，将为企业与经济体提供新的机遇。
例如，传统的统计方法采用历史数据收集技术，这种方式通常将数据、分析和决策分隔开来。
伴随着先进的系统监控和信息技术成本的下降，工作能力大大提高，
实时数据处理的规模得以大大提升，高频率的实时数据为系统操作提供全新视野。
机器分析则为分析流程开辟新维度，各种物理方式之结合、行业特定领域的专业知识、
信息流的自动化与预测能力相互结合可与现有的整套“大数据”工具联手合作。
最终，工业互联网将涵盖传统方式与新的混合方式，通过先进的特定行业分析，充分利用历史与实时数据。
"""

def profile_ime_generate():
    start_time = time.time()
    
    print("profile generate with context length = 3")
    for i in range(10):
        candidates = ime.generate("你好 friends ", "zaijian")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds") # 1.7s

    print("profile generate with context length = 0")
    start_time = time.time()
    
    for i in range(10):
        candidates = ime.generate("", "zai")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds") #0.96

    print("profile generate with context length = 10")
    start_time = time.time()
    
    for i in range(10):
        candidates = ime.generate("今天下午的天气看上去", "henbucuo")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds") #3.2


def produce_consecutive_miu_inputs(sentence):
    context_pinyin_pairs = []
    k = 0
    while k < len(sentence):
        i = k
        while i < len(sentence) and is_chinese_char(sentence[i]):
            i+=1
        while i != k:
            len_miu = i - k
            j = random.randint(0, len_miu)
            context = sentence[:k]
            target = sentence[k:k+j]
            target_pinyin = lazy_pinyin(sentence[k:k+j])
            pinyin_str = "".join(target_pinyin)
            for p in range(1,len(pinyin_str) +1):
                prefix = pinyin_str[:p]
                context_pinyin_pairs.append((context, prefix))
                # print(context, prefix) # replace to
            k = k + j
        while k < len(sentence) and not is_chinese_char(sentence[k]):
            k += 1
    return context_pinyin_pairs


def _profile_discontinuous_typing(engine, sentence): # 14s
    print(f"profile on {sentence}")
    start_time = time.time()
  
    context_pinyin_pairs = produce_consecutive_miu_inputs(sentence)
    for context, prefix in context_pinyin_pairs:
        print(context, prefix)
        candidates = engine.generate(context, prefix)

    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time} seconds") # 1.7s


def profile_ime_continuous_typing(): # 16s
    sentence = demo.strip().split("\n")[1]
    _profile_discontinuous_typing(ime, sentence)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a function to profile")
    parser.add_argument("--opt", choices=["ime_gen", "ime_typing"], help="Select the function to profile")

    args = parser.parse_args()

    if args.opt == "ime_gen":
        profile_ime_generate()
    elif args.opt == "ime_typing":
        profile_ime_continuous_typing()