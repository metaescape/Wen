import torch
from transformers import GPT2LMHeadModel, BertTokenizer, LogitsProcessorList
from wen.generation import (
    beam_search,
    PinyinGPTCompatibleLogitsProcessor,
    new_prepare_inputs_for_generation,
    generate,
)
from wen.pinyin import get_pinyin_to_char
from typing import Union
from pinyinsplit import PinyinSplit
from functools import lru_cache
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple


@dataclass
class GenerationOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using Beamsearch.


    Args:
        candidates_list (`Tuple[list[str]]`)
            Tuple of generated token Tuple for each beam;
        input_ids (`torch.LongTensor` of shape `(num_beams, sequence_length)`)
            input token ids for each beam.
        pkv (`Optional[list[list[torch.FloatTensor]]]` of length `num_layers`)
            Tuple of the past key value tuples used by the model for each beam.
            Each tuple has a shape of `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`
        beam_scores_list (`Tuple[torch.FloatTensor]` of shape `(num_beams)`)
            Tuple of the scores for each step for each beam.

    """

    candidates_list: Optional[Tuple[Tuple[str]]] = None
    input_ids: Optional[torch.LongTensor] = None
    pkv: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    beam_scores_list: Optional[Tuple[torch.FloatTensor]] = None


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
        GPT2LMHeadModel.generate = generate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(
            device
        )
        self.context_length = 10
        self.num_return_sequences = 9
        self.num_beams = 10
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.processors = LogitsProcessorList()
        self.pinyin_dict = get_pinyin_to_char(self.tokenizer, pinyin_json)
        self.processors.append(
            PinyinGPTCompatibleLogitsProcessor(self.pinyin_dict)
        )
        self.pys = PinyinSplit()
        self.past_key_values_pools = {}
        self.recent_key_values = None
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

    def generate(self, context, pinyin: Union[str, list], logger=None) -> list:
        context = context[-self.context_length :]

        context_key_values, context = self.cls_context(context)

        with_cls = True if context == "" else False
        context_ids = self.prepare_context_ids(context, with_cls=with_cls)
        constraint_pinyin_list = (
            pinyin if isinstance(pinyin, list) else self.split(pinyin)
        )
        if constraint_pinyin_list == []:
            return []

        context_len = context_ids.size(1)
        max_length = context_len + len(constraint_pinyin_list)

        # if context in self.past_key_values_pools:
        #     past_key_values = self.past_key_values_pools[context]
        output_ids, last_key_values, beam_history = self.model.generate(
            input_ids=context_ids,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            logits_processor=self.processors,
            max_length=max_length,
            constraint_pinyin_list=constraint_pinyin_list,
            past_key_values=context_key_values,
        )
        res = []
        candidates_list = []
        output_ids = output_ids[:, context_len:]
        if logger and last_key_values:
            logger.debug(last_key_values.size())
        for i in range(output_ids.shape[0]):
            token_list = self.tokenizer.convert_ids_to_tokens(output_ids[i])
            res.append("".join(token_list))
            candidates_list.append(tuple(token_list))
        self.last = GenerationOutput(
            candidates_list=tuple(candidates_list),
            input_ids=output_ids,
            pkv=last_key_values,
            beam_scores_list=beam_history,
        )
        return res

    def create_context_pkv(self, context) -> tuple:
        """concatenate the past_key_values of oth the CLS token and the pure context
        most of the time, just retren self.recent_key_values (already concatenated)

        but we need to keep the length of the past_key_values not exceeding self.context_length + 1

        return (past_key_values, remaining_context)
        """

        if self.recent_key_values:
            length = self.recent_key_values[0][0].size(2)
            if length > self.context_length + 1:
                new_key_values = []
                for key, value in self.recent_key_values:
                    new_key_values.append(
                        (
                            key[:, :, -self.context_length - 1 :, :],
                            value[:, :, -self.context_length - 1 :, :],
                        )
                    )
                self.recent_key_values = new_key_values

            last_context = context[-1] if context else context
            return self.recent_key_values, context

        else:
            return self.cls_kv, context

    def cls_context(self, context):
        return self.cls_kv, context

    def lru_context(self, context):
        if context in self.past_key_values_pools:
            return self.past_key_values_pools[context], context
        else:
            return self.cls_kv, context
