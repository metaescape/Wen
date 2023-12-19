import torch
from transformers import GPT2LMHeadModel, BertTokenizer, LogitsProcessorList
from wen.generation import (
    beam_search,
    PinyinGPTCompatibleLogitsProcessor,
    new_prepare_inputs_for_generation,
    generate,
    _expand_inputs_for_generation,
)
from wen.pinyin import get_pinyin_to_char, PinYinTokenizer
from typing import Union
from pinyinsplit import PinyinSplit
from functools import lru_cache
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from collections import deque
import json


@dataclass
class CacheItem:
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

    context_key: Tuple[str, ...] = ()
    code_key: Tuple[str, ...] = ()
    candidates_list: Optional[Tuple[Tuple[str]]] = None
    input_ids: Optional[torch.LongTensor] = None
    pkv: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    beam_scores_list: Optional[Tuple[torch.FloatTensor]] = None

    def match_key(self, context_query: tuple, code_query: tuple):
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
                self.beam_scores_list[-1],
            )

        num_beams = self.input_ids.size(0)
        candidates, input_ids, pkv, beam_scores = None, None, None, None

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
                break

        return candidates, input_ids, pkv, beam_scores

    def candidate_list_match(self, context_query: tuple):
        for candidate in self.candidates_list:
            length, end  = longest_substring_to_suffix_match(candidate, context_query)
                
            if candidate == context:
                return True 


    def code_prefix_match(self, code_query: tuple):
        """
        return the score of matching the code_query with the code_key
        """
        return longest_common_prefix_length(self.code_key, code_query)

    def context_match(self, context_query: tuple):
        return longest_substring_to_suffix_match(
            self.context_key, context_query
        )


def longest_substring_to_suffix_match(
    context_key: tuple, context_query: tuple
):
    """
    context_key 的后缀必须完全匹配 context_query 的某个子序列
    返回最长匹配长度和最长匹配子串在 context_query 中的结束位置

    >>> longest_substring_to_suffix_match((1,2,3,4,5), (3,4,5,7))
    (3, 2)
    """
    len_key, len_query = len(context_key), len(context_query)
    max_match_length = 0
    max_match_end_index = -1

    for query_end in range(len_query):
        match_length = 0

        for key_end in range(min(len_key, len_query - query_end)):
            if (
                context_query[-query_end - 1 - key_end]
                == context_key[-key_end - 1]
            ):
                match_length += 1
                if match_length > max_match_length:
                    max_match_length = match_length
                    max_match_end_index = len_query - query_end - 1
            else:
                break

    return max_match_length, max_match_end_index


def longest_common_prefix_length(tuple_a, tuple_b):
    prefix_length = 0
    for a, b in zip(tuple_a, tuple_b):
        if a == b:
            prefix_length += 1
        else:
            break
    return prefix_length

class IME:
    """
    A input method engine base class without history cache
    """
    def __init__(
        self,
        model_name_or_path="aihijo/gpt2-zh-21k",
        pinyin_json="wen/data/pinyin2char.json",
    ) -> None:
        self.model = self.init_model(model_name_or_path)

        self.max_context_length = 10
        self.num_return_candidates = 9
        self.num_beams = 10

        with open(pinyin_json) as f:
            self.pinyin2char = json.load(f)

        self.code_tokenizer = PinYinTokenizer(pinyin2char=self.pinyin2char)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.processors = LogitsProcessorList()
        self.code_map = get_pinyin_to_char(self.tokenizer, self.pinyin2char)
        self.processors.append(
            PinyinGPTCompatibleLogitsProcessor(self.code_map)
        )
        self.pys = PinyinSplit()

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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(
            self.device
        )
        model.eval()
        return model

    def init_cls_kv(self):
        """
        get the past_key_values of the CLS token
        this is used to concatenate with the past_key_values of the pure context
        """
        cls_ids = torch.tensor([[101]])
        self.cls_kv = self.model(cls_ids).past_key_values


    def prepare_context_ids(self, context) -> torch.Tensor:
        start = 0 if context == "" else 1
        context_ids = self.tokenizer.encode(context)[start:-1]
        input_ids = (
            torch.Tensor(context_ids).long().unsqueeze(0).to(self.model.device)
        )
        return input_ids

    def split(self, code_string: str):
        code_tokens = self.pys.split(code_string)
        return code_tokens[0] if code_tokens else []

    def generate(
        self, context_string: str, code_string: str, logger=None
    ) -> list:
        """
        generate candidates without pkv optmization
        """
        context_ids_list = [101]

        if context_string.strip() != "":
            (
                token_list,
                pinyin_list,
                pinyin_id_list,
                input_ids_list,
            ) = self.code_tokenizer.encode_string_list([context_string], self.tokenizer)
            
            context_ids_list += input_ids_list

        current_code_tokens = self.split(code_string)

        context_ids = (
                torch.Tensor(context_ids_list).long().unsqueeze(0).to(self.model.device)
            )
        context_len = context_ids.size(1)
        max_length = len(context_ids_list) + len(current_code_tokens)

        output_ids, last_key_values, beam_history = self.model.generate(
                input_ids=context_ids,
                beam_scores=None,
                num_beams=16,
                num_return_sequences=10,
                logits_processor=self.processors,
                max_length=max_length,
                constraint_pinyin_list=current_code_tokens,
                past_key_values=None,
            )
    
        res = []
        candidates_list = []
        target_ids = output_ids[:, context_len:]

        for i in range(target_ids.shape[0]):
            token_list = self.tokenizer.convert_ids_to_tokens(target_ids[i])
            res.append("".join(token_list))
            candidates_list.append(tuple(token_list))

        return res[: self.num_return_candidates]
    

class TypinGPT(IME):
    def __init__(
        self,
        model_name_or_path="aihijo/gpt2-zh-21k",
        pinyin_json="wen/data/pinyin2char.json",
    ) -> None:
        
        super().__init__(model_name_or_path, pinyin_json)

        self.max_history_length = 30

        self.init_beam_scores()


        self.history = deque(maxlen=self.max_history_length)
        self.cls_kv = []

    def init_beam_scores(self):
        beam_scores = torch.zeros(
            (1, self.num_beams),
            dtype=torch.float,
            device=self.device,
        )
        beam_scores[:, 1:] = -1e9
        self.beam_scores = beam_scores.view((self.num_beams,))


    def generate(
        self, context_string: str, code_string: str, logger=None
    ) -> list:
        context_query, context_ids = self.prepare_context(context_string)

        code_query = self.prepare_code_query(code_string)

        if code_query == ():
            # not a valid coding string, e.g "haope" in pinyin
            return []

        query = context_query + code_query

        (
            cache_candidates,
            cache_context_ids,
            pkv,
            beam_scores,
        ) = self.search_history(query)

        if cache_context_ids is not None:
            context_ids = cache_context_ids[:, -1:]

        pin_context_length = len(code_query)

        if logger and context_query:
            logger.debug(
                f"{context_query}, {code_query}, {current_code_tokens}"
            )

        if current_code_tokens == tuple():
            logger.debug("current_code_tokens is empty, just return")
            return [  # 重新输入的情况，比如选错了词，删除后重新输入
                "".join(cache_candidates[i][-pin_context_length:])
                for i in range(len(cache_candidates))
            ]

        context_len = context_ids.size(1)
        max_length = context_len + len(current_code_tokens)

        output_ids, last_key_values, beam_history = self.model.generate(
            input_ids=context_ids,
            beam_scores=beam_scores
            if beam_scores is not None
            else self.beam_scores,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            logits_processor=self.processors,
            max_length=max_length,
            constraint_pinyin_list=current_code_tokens,
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
            (cache_context_ids.expand(target_ids.size(0), -1), target_ids),
            dim=1,
        )

        if len(cache_candidates) > 1:
            candidates_list = [
                _b + item
                for _b, item in zip(cache_candidates, candidates_list)
            ]

        key = context_query + tuple(current_code_tokens)
        last_cache = CacheItem(
            key=key,
            candidates_list=tuple(candidates_list),
            input_ids=output_ids,
            pkv=last_key_values,
            beam_scores_list=beam_history,
        )
        self.history.append(last_cache)

        # if pinyin == "mapengyou":  # test debug
        #     breakpoint()

        if pin_context_length > 0:
            res = [
                "".join(cache_candidates[i][-pin_context_length:]) + res[i]
                for i in range(len(res))
            ]

        return res[: self.num_return_candidates]

    def prepare_context(
        self, context_string: str
    ) -> tuple[tuple[str], torch.Tensor]:
        context_query = tuple()
        context_ids = [101]
        if context_string != "":
            (
                context_query,
                _pinyin_list,
                _pinyin_id_list,
                context_ids_without_cls,
            ) = self.code_tokenizer.encode_string_list(
                [context_string], self.tokenizer
            )
            context_query = tuple(context_query[-self.max_context_length :])
            context_ids = context_ids + context_ids_without_cls

        context_ids = (
            torch.Tensor(context_ids).long().unsqueeze(0).to(self.device)
        )

        return context_query, context_ids

    def prepare_code_query(self, code_string: str) -> tuple:
        code_query = tuple(self.split(code_string))

        return code_query

    def prepare_for_generation(self, context_string: str, code_string: str):
        pass


    def search_history(self, query):
        for i in range(len(self.history) - 1, -1, -1):
            candidates, input_ids, pkv, beam_scores = self.history[
                i
            ].match_key(query)
            if pkv is not None:
                return candidates, input_ids, pkv, beam_scores

        return tuple(), None, None, None

    def clear_history(self):
        self.history.clear()
