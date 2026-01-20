# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Tuple

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceStatus
from vllm.transformers_utils.tokenizer import AnyTokenizer


class StopChecker:
    """LLMEngine helper class which separates out the logic involving stop
    checking. This checks things such as: whether the eos token was emitted,
    whether the max_tokens has been consumed, whether a stop string has been
    emitted, or if we have exceeded the max model len.
    """

    def __init__(self, max_model_len: int,
                 get_tokenizer_for_seq: Callable[[Sequence], AnyTokenizer]):
        # Do not use it directly, but use `self._get_max_model_len`.
        self._max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_tokenizer_for_seq

    def _get_max_model_len(self, lora_req: Optional[LoRARequest]):
        if lora_req and lora_req.long_lora_max_len:
            return lora_req.long_lora_max_len
        else:
            return self._max_model_len

    def maybe_stop_sequence(
        self,
        seq: Sequence,
        new_char_count: int,
        sampling_params: SamplingParams,
        lora_req: Optional[LoRARequest] = None,
        is_summary_stage: bool = False,
        summary_token_id: Optional[int] = None,
    ) -> None:
        """Stop the finished sequences.

       new_char_count is the number of chars added to the
           sequence's output text for the newly generated token
       
       summary_token_id: The token ID for <summary>, used to determine the start of summary stage.
           If None, summary stage handling is skipped.
        """

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            raise ValueError("Unreachable code reached -- ignore_eos should be True")
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = seq.get_last_token_id()
        # Calculate summary_end_token_id as (summary_token_id + 1) for paired tokens like <summary>/</summary>
        summary_end_token_id = (summary_token_id + 1) if summary_token_id is not None else None
        if last_token_id in (sampling_params.stop_token_ids or ()):
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                seq.output_text = seq.output_text[:-new_char_count]
            if is_summary_stage and summary_end_token_id is not None:
                seq.status = SequenceStatus.FINISHED_STOPPED
                seq.data._output_token_ids[-1] = summary_end_token_id
                seq.data._new_appended_tokens[-1] = summary_end_token_id
                seq.data._cached_all_token_ids[-1] = summary_end_token_id
                seq.stop_reason = summary_end_token_id
            elif len(seq.get_output_token_ids()) % seq.block_size == 0:
                seq.status = SequenceStatus.FINISHED_STOPPED
                seq.stop_reason = last_token_id
            else:
                seq.is_padding = True
            return
    
        # TODO(syf) Add a description for why `is_padding` is necessary when using PagedAttention for ParaThinker
        if seq.is_padding and (len(seq.get_output_token_ids()) % seq.block_size == 0):
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.is_padding = False
            seq.stop_reason = last_token_id

        # Check if any stop strings are matched.
        stop = self.check_stop_strings(
            seq.output_text, new_char_count, sampling_params.stop,
            sampling_params.include_stop_str_in_output)
        if stop is not None:
            raise RuntimeError("UNREACHABLE")
            stop_str, truncate_to = stop
            if truncate_to != -1:
                seq.output_text = seq.output_text[:truncate_to]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_str
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self._get_max_model_len(lora_req):
            print(f"WARNING: Exceed max model length.")
            print(f"sequence_len={seq.get_len()}, max_model_len={self._get_max_model_len(lora_req)}")
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if ((not is_summary_stage) and seq.get_output_len() == sampling_params.max_tokens):
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return
        
        # Locate the position of <summary> token to get the precise summary length
        if is_summary_stage and summary_token_id is not None:
            total_len = seq.get_output_len()
            output_token_ids = list(seq.data._output_token_ids)
            if summary_token_id in output_token_ids:
                summary_start_idx = output_token_ids.index(summary_token_id)
                if (total_len - summary_start_idx) == sampling_params.summary_max_tokens:
                    seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED

    @staticmethod
    def check_stop_strings(
        output_text: str,
        new_char_count: int,
        stop: List[str],
        include_in_output: bool,
    ) -> Optional[Tuple[str, int]]:
        """Check if any stop strings are matched and truncate sequence
        output text accordingly.

        Returns tuple (stop_string, offset) if matched or else None.

        Where stop_string is the matched stop string and offset is the
        length to which output_text should be truncated, or -1 for no
        truncation.
        """
        if not new_char_count or not stop:
            return None

        for stop_str in stop:
            stop_string_len = len(stop_str)
            # Avoid searching already-searched text.
            stop_index = output_text.find(stop_str,
                                          1 - new_char_count - stop_string_len)
            if stop_index == -1:
                continue

            if include_in_output:
                # Truncate to end of stop string.
                stop_index += stop_string_len
                if stop_index >= len(output_text):
                    # No truncation required.
                    return stop_str, -1

            # Truncate the output text to either the beginning
            # or end of the stop string.
            return stop_str, stop_index
        return None
