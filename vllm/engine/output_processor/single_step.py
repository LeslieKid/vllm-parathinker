# SPDX-License-Identifier: Apache-2.0

import math
from typing import List

from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sequence import (CompletionSequenceGroupOutput, SequenceGroup,
                           SequenceGroupOutput, SequenceStage, SequenceStatus, Logprob)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)


def single_step_process_prompt_logprob(
        sg_output_proc: SequenceGroupOutputProcessor, seq_group: SequenceGroup,
        output: CompletionSequenceGroupOutput) -> None:
    """Process prompt logprobs associated with the :class:`SequenceGroupOutput`
    for a given step.

    Do nothing if the output has no prompt logprobs.

    Account for the fact that transformers do not compute first-token logprobs.
    
    Args:
      sg_output_proc: :class:`SequenceGroupOutputProcessor` instance
      seq_group: the output is associated with this :class:`SequenceGroup`
      output: the :class:`SequenceGroupOutput` for a single scheduler step
    """
    prompt_logprobs = output.prompt_logprobs

    # If this is the first (or only) "chunk" of the prefill, we need
    # to prepend None to the list of prompt logprobs. The reason for this
    # is that for N prompt tokens, the Sampler will generate N-1 total
    # prompt logprobs during prefill since the token at idx 0 will not
    # have a logprob associated with it.
    if prompt_logprobs is not None:
        if not seq_group.prompt_logprobs:
            prompt_logprobs = [None] + prompt_logprobs
            seq_group.prompt_logprobs = []

        assert hasattr(sg_output_proc, 'detokenizer')
        if (seq_group.sampling_params.detokenize
                and sg_output_proc.detokenizer):
            sg_output_proc.detokenizer.decode_prompt_logprobs_inplace(
                seq_group,
                prompt_logprobs,
                position_offset=len(seq_group.prompt_logprobs))

        seq_group.prompt_logprobs.extend(prompt_logprobs)


class SingleStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles "output processing" logic,
    which happens after the model returns generated token ids and before
    scheduling of the next batch. Output processing logic includes
    detokenization, and determining if a sequence is finished (e.g. via max len
    or eos token).

    The SingleStepOutputProcessor is specialized to the case where the model
    emits at most a single token per invocation, which precludes configurations
    such as speculative decoding or multi-step decoding. This enables beam
    search sampling, which requires forking/finishing/freeing sequences in a way
    that is currently difficult to schedule multiple steps ahead of time.
    """

    def __init__(self, scheduler_config: SchedulerConfig,
                 detokenizer: Detokenizer, scheduler: List[Scheduler],
                 seq_counter: Counter, stop_checker: StopChecker):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.stop_checker = stop_checker

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput],
                        cot_token_ids: List[int],
                        okay_token_ids: List[List[int]],
                        summary_token_ids: List[int],
                        parthink_size: int,
                        pad_token_id: int,
                        is_async: bool) -> None:
        """Append all new tokens to sequences in the sequence group. Fork any
        surviving beam candidates; free any unsurviving ones.

        Invokes detokenizer to detokenize new tokens, and also marks sequences
        as finished if they meet stop conditions.
        
        is_async - Indicates whether this postprocessor runs in 
            parallel with the GPU forward pass and is processing 
            tokens from the previous step. If this is true, then
            no tokens need to be appended since it is already done
            externally (before the next schedule() call)
        """
        assert (len(outputs) == 1
                ), f"{type(self)} does not support multiple outputs per step"
        return self._process_sequence_group_outputs(sequence_group, outputs[0],
                                                    cot_token_ids, okay_token_ids, summary_token_ids, parthink_size, pad_token_id, is_async)

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        """Process prompt logprobs associated with one step of a single-step-
        scheduled computation.
        
        Args:
          seq_group: the output is associated with this :class:`SequenceGroup`
          outputs: the :class:`SequenceGroupOutput` for a single scheduler step
        """
        assert len(outputs) == 1, "Single step should only have 1 output."
        output = outputs[0]
        assert isinstance(output, CompletionSequenceGroupOutput)
        single_step_process_prompt_logprob(self, seq_group, output)

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput,
                                        cot_token_ids: List[int],
                                        okay_token_ids: List[List[int]],
                                        summary_token_ids: List[int],
                                        parthink_size: int,
                                        pad_token_id: int,
                                        is_async: bool) -> None:
        # TODO(syf) This important function uses token id for specific tokenizer, leading to bad generalization. Fix it.
        think_token_id = 151648 # token id for `<think>`
        sampling_params = seq_group.sampling_params
        custom_token_probs = 0.99

        # Fork prompt multiple times after prefill phase for further parallel thinking (reuse KV caches for prompt)
        if (not seq_group.is_think_stage_finished()) and seq_group.num_seqs() == 1 and (not seq_group.first_seq.is_prefill()):
            source_seq = seq_group.first_seq
            
            if source_seq.get_output_len() == 0:
                # Create additional (parthink_size-1) sequence in the sequence group with sequence forking
                for i in range(parthink_size-1):
                    new_seq_id = next(self.seq_counter)
                    new_seq = source_seq.fork(new_seq_id=new_seq_id)
                    
                    # Register the forked sequence with ALL schedulers in the list
                    for scheduler in self.scheduler:
                        scheduler.fork_seq(parent_seq=source_seq, child_seq=new_seq)
                    
                    seq_group.seqs.append(new_seq)
                    seq_group.seqs_dict[new_seq_id] = new_seq
                    
                seq_group.is_single_seq = len(seq_group.seqs) == 1
                assert seq_group.num_seqs() == parthink_size
                
                # Custom special tokens IDs (e.g. <think1>, <think2>) as first sample tokens IDs
                assert len(cot_token_ids) >= parthink_size
                samples = cot_token_ids[:parthink_size]
                if not is_async:
                    seqs = seq_group.get_seqs(status=None)
                    for i in range(parthink_size):
                        logprobs = {samples[i]:Logprob(logprob=math.log(custom_token_probs))}
                        seqs[i].append_token_id(samples[i], logprobs=logprobs)
                        
                return None
            else:
                raise RuntimeError("The sequence should not generate any output")
        
        # Parallel thinking: think / reasoning stage
        if not seq_group.is_think_stage_finished():
            # Important: The loop process sequences one by one but the is_think_stage_finished() should consider the 
            # entire sequence group. ps: Two separated loops & From StopChecker to ThinkStageChecker
            for seq in seq_group.seqs:
                if sampling_params.detokenize and self.detokenizer:
                    new_char_count = self.detokenizer.decode_sequence_inplace(
                        seq, sampling_params)
                else:
                    new_char_count = 0
                self.stop_checker.maybe_stop_sequence(
                    seq,
                    new_char_count,
                    sampling_params,
                    lora_req=seq_group.lora_request,
                )
                
            if not is_async and (not seq_group.is_think_stage_finished()):
                for (sample, seq) in zip(outputs.samples, seq_group.get_unfinished_seqs()):
                    sample_output_token = sample.output_token
                    sample_logprobs = sample.logprobs
                    
                    # Parallel thinking stage should not generate the special cot start token ids,
                    # if it occurs by accident (because of the randomness of sampling), replace it with `<think>` directly.
                    if (sample_output_token in cot_token_ids) or (sample_output_token == summary_token_ids[0]):
                        sample_output_token = think_token_id
                        sample_logprobs = {sample_output_token:Logprob(logprob=math.log(1-custom_token_probs))}
                        
                    # Check whether the current reasoning path encounter stop tokens
                    # Match the end tokens (e.g. </think1>, </think2>) for parallel thinking stage
                    if len(seq.data.output_token_ids) > 0 and sample_output_token in (sampling_params.stop_token_ids or ()):
                        first_output_token_id = seq.data.output_token_ids[0]
                        assert first_output_token_id >= 151665 and first_output_token_id <= 151679 and first_output_token_id % 2 == 1
                        end_token_id = first_output_token_id + 1
                        sample_output_token = end_token_id
                        sample_logprobs = {sample_output_token:Logprob(logprob=math.log(custom_token_probs))}
                    
                    offset = seq.get_output_len() - 1
                    # `okay_list` forces inference engine output specific tokens at the very begining of the reasoning
                    # path. Empty by default.
                    okay_list_idx = (seq.data.output_token_ids[0] - 151665) // 2
                    if offset < len(okay_token_ids[okay_list_idx]):
                        sample_output_token = okay_token_ids[okay_list_idx][offset]
                        sample_logprobs = {sample_output_token:Logprob(logprob=math.log(custom_token_probs))}
                    
                    (appended_token_id, logprobs) = (pad_token_id, {pad_token_id:Logprob(logprob=math.log(custom_token_probs))}) \
                                                    if seq.is_padding == True \
                                                    else (sample_output_token, sample_logprobs)
                    seq.append_token_id(appended_token_id, logprobs)

        if seq_group.is_think_stage_finished():
            # Think stage ended just now (think stage -> summary stage)
            if seq_group.num_seqs() == parthink_size:
                # Set all reasoning paths as Finished
                for seq in seq_group.seqs:
                    seq.status = SequenceStatus.FINISHED_STOPPED
                new_seq_id = next(self.seq_counter)
                new_seq = seq_group.prepare_merged_sequence(new_seq_id)
                new_seq.data._stage = SequenceStage.DECODE
                new_seq.status = SequenceStatus.RUNNING
                assert len(self.scheduler) == 1
                for scheduler in self.scheduler:
                    scheduler.combine_seqs(seq_group, new_seq)
                # Remove original sequences used in think stage
                for seq in seq_group.seqs:
                    if seq.is_finished():
                        for scheduler in self.scheduler:
                            scheduler.free_seq(seq)
                # Determine that there is no summary token in the new sequence (Avoid encountering bug in the following `index()`)
                assert new_seq.get_output_token_ids().count(summary_token_ids[0]) == 0
                # Add new sequence for summary stage
                seq_group.seqs.append(new_seq)
                seq_group.seqs_dict[new_seq_id] = new_seq
                seq_group.is_single_seq = len(seq_group.seqs) == 1
                
                # Custom special token id <summary> as the first token id in summary stage
                sample = summary_token_ids[0]
                if not is_async:
                    logprobs = {sample:Logprob(logprob=math.log(custom_token_probs))}
                    seq_group.get_seqs()[-1].append_token_id(sample, logprobs=logprobs)
                    
                # Update num_computed_tokens
                seq_group.update_num_computed_tokens(seq_group.get_seqs()[-1].get_len() - 1)
                return None
        
            # Summary stage
            else:
                assert len(outputs.samples) == 1
                seq = seq_group.seqs[-1]
                assert (not is_async)
                if not is_async:
                    # `summary_token_ids` can be considered as specific template for summary stage
                    summary_token_idx = seq.get_output_token_ids().index(summary_token_ids[0])
                    offset = seq.get_output_len() - summary_token_idx
                    if offset < len(summary_token_ids):
                        output_token = summary_token_ids[offset]
                        logprobs = {output_token:Logprob(logprob=math.log(custom_token_probs))}
                        seq.append_token_id(output_token, logprobs)
                    else:
                        sample = outputs.samples[0]
                        seq.append_token_id(sample.output_token, sample.logprobs)
                if sampling_params.detokenize and self.detokenizer:
                    new_char_count = self.detokenizer.decode_sequence_inplace(
                        seq, sampling_params)
                else:
                    new_char_count = 0
                self.stop_checker.maybe_stop_sequence(
                    seq,
                    new_char_count,
                    sampling_params,
                    lora_req=seq_group.lora_request,
                    is_summary_stage=True,
                )
                if seq.is_finished():
                    for scheduler in self.scheduler:
                        scheduler.free_seq(seq)
        