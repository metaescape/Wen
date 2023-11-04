import torch
from transformers import GPT2LMHeadModel, BertTokenizer
from generation import beam_search, PinyinGPTCompatibleLogitsProcessor
from transformers import LogitsProcessorList
from pinyin import get_pinyin_to_char
from typing import Union
from pinyinsplit import PinyinSplit
from functools import lru_cache


class TypinGPT:
    def __init__(
        self,
        model_name_or_path="aihijo/gpt2-zh-21k",
        pinyin_json="data/pinyin2char.json",
    ) -> None:
        GPT2LMHeadModel.beam_search = beam_search
        # deprecate params validation
        GPT2LMHeadModel._validate_model_kwargs = lambda self, model_kwargs: 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(
            device
        )
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.processors = LogitsProcessorList()
        self.pinyin_dict = get_pinyin_to_char(self.tokenizer, pinyin_json)
        self.processors.append(
            PinyinGPTCompatibleLogitsProcessor(self.pinyin_dict)
        )
        self.pys = PinyinSplit()
        self.past_key_values_pools = {}

    def split(self, pinyin: str):
        pinyin_list = self.pys.split(pinyin)
        return pinyin_list[0] if pinyin_list else []

    async def generate(self, context, pinyin: Union[str, list]):
        context_ids = self.tokenizer.encode_plus(context[-10:])["input_ids"][
            :-1
        ]
        constraint_pinyin_list = (
            pinyin if isinstance(pinyin, list) else self.split(pinyin)
        )
        if constraint_pinyin_list == []:
            return []

        if context in self.past_key_values_pools:
            past_key_values = self.past_key_values_pools[context]

        output_ids, past_key_values = self.model.generate(
            input_ids=torch.Tensor(context_ids)
            .long()
            .unsqueeze(0)
            .to(self.model.device),
            num_beams=10,
            num_return_sequences=9,
            logits_processor=self.processors,
            max_length=len(context_ids) + len(constraint_pinyin_list),
            constraint_pinyin_list=constraint_pinyin_list,
            # past_key_values=past_key_values,
        )
        res = []
        output_ids = output_ids[:, len(context_ids) :]
        for i in range(output_ids.shape[0]):
            res.append(self.tokenizer.decode(output_ids[i]).replace(" ", ""))
        return res
