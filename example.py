import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 获得指定文件夹的绝对路径，兼容所有操作系统，path就是个字符串
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 从指定的路径下面完成tokenizer的加载
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 根据路径完成LLM的加载，TP=1，默认enforce_eager=True
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
