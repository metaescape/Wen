import torch
from transformers import GPT2LMHeadModel, BertTokenizer, LogitsProcessorList
from wen.generation import (
    beam_search,
    PinyinGPTCompatibleLogitsProcessor,
    new_prepare_inputs_for_generation,
    generate,
    _expand_inputs_for_generation,
)
from wen.pinyin import get_pinyin_to_char
from typing import Union
from pinyinsplit import PinyinSplit
from functools import lru_cache
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from collections import deque


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

    key: Optional[Tuple[str]] = None
    candidates_list: Optional[Tuple[Tuple[str]]] = None
    input_ids: Optional[torch.LongTensor] = None
    pkv: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    beam_scores_list: Optional[Tuple[torch.FloatTensor]] = None

    def match_context(self, context: tuple):
        """
        match the candidates with the context
        there are two kind of matching:
        1. exact full context matching
        2. partial context matching

        all returned tensors are of shape (num_beams, ...), including exact full context matching
        """

        if self.key == context:
            return (
                self.candidates_list,
                self.input_ids,
                self.pkv,
                self.beam_scores_list,
            )

        candidates, input_ids, pkv, beam_history = None, None, None, None
        num_beams = self.input_ids.size(0)

        for i, candidates in enumerate(self.candidates_list):
            if candidates == context:
                # return pkv
                candidates = (self.candidates_list[i],) * num_beams
                pkv = tuple(
                    [
                        (
                            self.pkv[j][0][
                                i : i + 1, :, :, :
                            ].repeat_interleave(num_beams, 0),
                            self.pkv[j][1][
                                i : i + 1, :, :, :
                            ].repeat_interleave(num_beams, 0),
                        )
                        for j in range(len(self.pkv))
                    ]
                )
                input_ids = self.input_ids[i : i + 1, :].repeat_interleave(
                    num_beams, 0
                )
                beam_history = None
                break

        return candidates, input_ids, pkv, beam_history


class TypinGPT:
    def __init__(
        self,
        model_name_or_path="aihijo/gpt2-zh-21k",
        pinyin_json="wen/data/pinyin2char.json",
    ) -> None:
        self.model = self.init_model(model_name_or_path)

        self.context_length = 10
        self.num_return_candidates = 9
        self.num_beams = 10
        self.max_history_length = 30

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.processors = LogitsProcessorList()
        self.pinyin_dict = get_pinyin_to_char(self.tokenizer, pinyin_json)
        self.processors.append(
            PinyinGPTCompatibleLogitsProcessor(self.pinyin_dict)
        )
        self.pys = PinyinSplit()

        self.last = GenerationOutput()
        self.history = deque(maxlen=self.max_history_length)
        self.last_pinyin = ""
        self.cls_kv = []
        self.init_cls_kv()

    def init_model(self, model_name_or_path):
        GPT2LMHeadModel.beam_search = beam_search
        # deprecate params validation
        GPT2LMHeadModel._validate_model_kwargs = lambda self, model_kwargs: 1
        GPT2LMHeadModel.prepare_inputs_for_generation = (
            new_prepare_inputs_for_generation
        )
        GPT2LMHeadModel.generate = generate
        GPT2LMHeadModel._expand_inputs_for_generation = (
            _expand_inputs_for_generation
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        model.eval()
        return model

    def split(self, pinyin: str):
        pinyin_list = self.pys.split(pinyin)
        return pinyin_list[0] if pinyin_list else []

    def prepare_context_ids(self, context) -> torch.Tensor:
        start = 0 if context == "" else 1
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

        pin_context = ""
        current_pinyin = pinyin
        if pinyin.startswith(self.last_pinyin):
            combo = True
            # get the suffix
            current_pinyin = pinyin[len(self.last_pinyin) :]
            pin_context = pinyin[: len(self.last_pinyin)]

        context_query, pinyin_query = self.create_query(context, pin_context)
        query = context_query + pinyin_query

        background, full_context_ids, pkv, beam_history = self.search_history(
            query
        )

        if full_context_ids is None:
            context_ids = self.prepare_context_ids(context)
            full_context_ids = context_ids
        else:
            context_ids = full_context_ids[:, -1:]

        constraint_pinyin_list = self.split(current_pinyin)
        if constraint_pinyin_list == []:
            return []

        context_len = context_ids.size(1)
        max_length = context_len + len(constraint_pinyin_list)

        if logger:
            logger.debug(background)

        output_ids, last_key_values, beam_history = self.model.generate(
            input_ids=context_ids,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            logits_processor=self.processors,
            max_length=max_length,
            constraint_pinyin_list=constraint_pinyin_list,
            past_key_values=pkv,
        )

        res = []
        candidates_list = []
        target_ids = output_ids[:, context_len:]

        for i in range(target_ids.shape[0]):
            token_list = self.tokenizer.convert_ids_to_tokens(target_ids[i])
            res.append("".join(token_list))
            candidates_list.append(tuple(token_list))

        output_ids = torch.cat(
            (full_context_ids.expand(target_ids.size(0), -1), target_ids),
            dim=1,
        )

        if len(background) > 1:
            candidates_list = [
                _b + item for _b, item in zip(background, candidates_list)
            ]

        key = context_query + tuple(constraint_pinyin_list)
        self.last = GenerationOutput(
            key=key,
            candidates_list=tuple(candidates_list),
            input_ids=output_ids,
            pkv=last_key_values,
            beam_scores_list=beam_history,
        )
        self.last_pinyin = pinyin
        self.history.append(self.last)

        # if pinyin == "mapengyou":  # test debug
        #     breakpoint()

        pin_context_length = len(pinyin_query)
        if pin_context_length > 0:
            res = [
                "".join(background[i][-pin_context_length:]) + res[i]
                for i in range(len(res))
            ]

        return res[: self.num_return_candidates]

    def create_query(
        self, context: str, partial_context: Optional[str] = None
    ):
        """
        create the query for search_history, there are two parts:

        1. context_query: the non pinyin context
        2. partial_context_query: the pinyin context
        """
        context_tokens = self.tokenizer.tokenize(context)
        context_query = tuple(context_tokens)
        partial_context_query = (
            tuple(self.split(partial_context))
            if partial_context is not None
            else tuple()
        )
        return context_query, partial_context_query

    def search_history(self, context_query, pinyin=None):
        for i in range(len(self.history)):
            candidates, input_ids, pkv, beam_history = self.history[
                i
            ].match_context(context_query)
            if pkv is not None:
                return candidates, input_ids, pkv, beam_history
        return tuple(), None, None, None
