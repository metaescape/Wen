
from wen.gpt import IME
from transformers import logging
logging.set_verbosity_error()


engine = IME(
    model_name_or_path="/data/database/models/gpt2-zh-21k",
    pinyin_json="wen/data/pinyin2char.json",
)

print("Model loaded")


def profile_generate():
    import time
    start_time = time.time()
    
    print("profile generate with context length = 3")
    for i in range(10):
        candidates = engine.generate("你好 friends ", "zaijian")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds") # 1.7s

    print("profile generate with context length = 0")
    start_time = time.time()
    
    for i in range(10):
        candidates = engine.generate("", "zai")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds") #0.96

    print("profile generate with context length = 10")
    start_time = time.time()
    
    for i in range(10):
        candidates = engine.generate("今天下午的天气看上去", "henbucuo")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds") #3.2

if __name__ == "__main__":
    profile_generate()