import torch
from transformers import GPT2LMHeadModel, BertTokenizer, LogitsProcessorList
from wen.generation import (
    beam_search,
    PinyinGPTCompatibleLogitsProcessor,
    new_prepare_inputs_for_generation,
)
from wen.pinyin import get_pinyin_to_char
from typing import Union
from pinyinsplit import PinyinSplit
from functools import lru_cache


class TypinGPT:
    def __init__(
        self,
        model_name_or_path="aihijo/gpt2-zh-21k",
        pinyin_json="wen/data/pinyin2char.json",
    ) -> None:
        GPT2LMHeadModel.beam_search = beam_search
        # deprecate params validation
        GPT2LMHeadModel._validate_model_kwargs = lambda self, model_kwargs: 1
        GPT2LMHeadModel.prepare_inputs_for_generation = (
            new_prepare_inputs_for_generation
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(
            device
        )
        self.context_length = 10
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.processors = LogitsProcessorList()
        self.pinyin_dict = get_pinyin_to_char(self.tokenizer, pinyin_json)
        self.processors.append(
            PinyinGPTCompatibleLogitsProcessor(self.pinyin_dict)
        )
        self.pys = PinyinSplit()
        self.past_key_values_pools = {}
        self.cls_kv = []
        self.init_cls_kv()

    def split(self, pinyin: str):
        pinyin_list = self.pys.split(pinyin)
        return pinyin_list[0] if pinyin_list else []

    def prepare_context_ids(self, context, with_cls=True) -> torch.Tensor:
        start = 0 if with_cls else 1
        context_ids = self.tokenizer.encode(context)[start:-1]
        input_ids = (
            torch.Tensor(context_ids).long().unsqueeze(0).to(self.model.device)
        )
        return input_ids

    def init_cls_kv(self):
        """
        get the past_key_values of the CLS token
        this is used to concatenate with the past_key_values of the pure context
        """
        cls_ids = self.prepare_context_ids("")
        self.cls_kv = self.model(cls_ids).past_key_values

    def generate(self, context, pinyin: Union[str, list]) -> list:
        context = context[-self.context_length :]
        context_ids = self.prepare_context_ids(context, with_cls=False)

        constraint_pinyin_list = (
            pinyin if isinstance(pinyin, list) else self.split(pinyin)
        )
        if constraint_pinyin_list == []:
            return []

        context_len = context_ids.size(1)
        max_length = context_len + len(constraint_pinyin_list)

        # if context in self.past_key_values_pools:
        #     past_key_values = self.past_key_values_pools[context]
        output_ids, self.past_key_values = self.model.generate(
            input_ids=context_ids,
            num_beams=10,
            num_return_sequences=9,
            logits_processor=self.processors,
            max_length=max_length,
            constraint_pinyin_list=constraint_pinyin_list,
            past_key_values=self.cls_kv,
        )
        res = []
        output_ids = output_ids[:, context_len:]
        for i in range(output_ids.shape[0]):
            res.append(self.tokenizer.decode(output_ids[i]).replace(" ", ""))
        return res
