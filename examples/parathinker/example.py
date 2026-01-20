from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import sys
import os
import warnings

os.environ["VLLM_USE_V1"] = "0" # V1 is not supported for ParaThinker
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_path = "Leslie04/ParaThinker-1.5B"

template = """<｜User｜>{question} You FIRST think about the reasoning process as an internal monologue \
    and then summarize the reasoning process to get the final answer. The summary process MUST BE enclosed \
    within <summary> </summary> tags. The final answer MUST BE put in \\boxed{{}}.<｜Assistant｜><think>"""

def load_math_problems(jsonl_path, start=1, end=30):
    math_problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i < start:
                continue
            if i > end:
                break
            data = json.loads(line)
            math_problems.append(data["problem"])
    return math_problems

dataset_file = "./inference/examples/parathinker/aime24.jsonl"
math_problems = load_math_problems(jsonl_path=dataset_file)

# Generate the full prompts using the template
prompts = [template.format(question=question) for question in math_problems]

block_size = 16 # default block_size in vllm
max_tokens = 1024 * 16
parthink_size = 4
assert max_tokens % block_size == 0

# Token IDs for </think>, </think1> ~ </think8>, </summary>
stop_token_ids = [151643, 151666, 151668, 151670, 151672, 151674, 151676, 151678, 151680, 151682, 151684]
sampling_params = SamplingParams(
    n=1,
    temperature=0.5,
    top_p=1.0,
    max_tokens=max_tokens,
    stop_token_ids=stop_token_ids,
    # Add eos token in stop_token_ids
    ignore_eos=True
)

# Append padding tokens
def prompts_preprocess(tokenizer, block_size) -> list[str]:
    processed_prompts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        length = len(tokens) + 1

        # Check if padding is needed
        if length % block_size != 0:
            num_pad = block_size - length % block_size
            padded_tokens = tokens + [tokenizer.pad_token_id] * num_pad
        else:
            padded_tokens = tokens

        # Decode the token list back to a string
        padded_prompt = tokenizer.decode(padded_tokens)
        processed_prompts.append(padded_prompt)

    return processed_prompts

def main():
    # Create an LLM.
    if max_tokens <= (8*1024):
        factor = 1.0
    else:
        factor = max_tokens / (8*1024)
    hf_overrides = {
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": 9280, # (max_prompt_len + max_reasoning_path_len + max_summary_len)
            "beta_fast": 32,
            "beta_slow": 1,
            "extrapolation_factor": 1.0,
            "attention_factor": 1.0, 
        }
    }
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        hf_overrides=hf_overrides,
        block_size=block_size,
        gpu_memory_utilization=0.9,
        dtype="float16"
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<vllm_pad>")
    tokenizer.pad_token = "<vllm_pad>"
    
    # Token IDs for parallel thinking tokens (<think1>, <think2>, etc.)
    # These can be obtained dynamically from the tokenizer for model-agnostic support:
    # For Qwen2.5: tokens are <think1> through <think8>
    # For other models: use the model-specific token names
    think_token_ids = []
    for i in range(1, 9):  # <think1> through <think8>
        token_name = f"<think{i}>"
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_name)
            if token_id != tokenizer.unk_token_id:
                think_token_ids.append(token_id)
            else:
                warnings.warn(f"Token '{token_name}' not found in tokenizer vocabulary (maps to unk_token)")
        except (AttributeError, KeyError, ValueError) as e:
            warnings.warn(f"Failed to get token ID for '{token_name}': {e}")
    
    # Fallback to hardcoded Qwen2.5 token IDs if dynamic lookup fails
    if not think_token_ids:
        warnings.warn("Using hardcoded Qwen2.5 token IDs as fallback. "
                     "For other models, ensure the tokenizer has <think1> through <think8> tokens.")
        think_token_ids = [151665, 151667, 151669, 151671, 151673, 151675, 151677, 151679]
    
    # The common tokens after think token
    okay_token_ids = [[]] * len(think_token_ids)
    # template for summary part
    summary_token_ids = tokenizer.encode(
        "<summary>By analyzing multiple reasoning processes above, I concluded that: The final answer is",
        add_special_tokens=False
    )

    prompts = prompts_preprocess(tokenizer, block_size)

    sys.setrecursionlimit(50000)
    outputs = llm.generate(
        cot_token_ids=think_token_ids,
        okay_token_ids=okay_token_ids,
        summary_token_ids=summary_token_ids,
        parthink_size=parthink_size,
        pad_token_id=int(tokenizer.pad_token_id),
        prompts=prompts,
        sampling_params=sampling_params
    )
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        print(f"Prompt:    {prompt!r}")
        idx = len(output.outputs) - 1
        generated_token_ids = output.outputs[idx].token_ids
        generated_text = tokenizer.decode(generated_token_ids)
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
