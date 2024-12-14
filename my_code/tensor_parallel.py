import sys
import time

# 将本地 vllm 代码目录添加到 sys.path 中
# sys.path.insert(0, "/home/yfman/vllm")  # 替换为你本地代码的实际路径

from vllm import LLM, SamplingParams
import os

weight_path = "/data0/yfman/hf_models/Llama-3-8B-Instruct"

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

llm = LLM(weight_path, tensor_parallel_size=4)
prompts = ["Where is the capital of France?"]
# 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

sum_time = 0
for i in range(10):
    # 记录当前时间
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)  # Generate texts from the prompts.
    end = time.time()
    sum_time += end - start
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("Average time: %f s" % (sum_time / 10))
