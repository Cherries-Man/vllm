from vllm import LLM, SamplingParams
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

weight_path = "/data1/yfman/hf_models/Llama-3.1-8B-Instruct"
# weight_path = "/data0/xiac/hf_models/Llama-3-8B-Instruct"
prompts = ["Where is the capital of France?"]  # Sample prompts.
# 输入的中文文本
# llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
llm = LLM(model=weight_path)  # Create an LLM.
print("\n llm.__class__.__dict__: %s \n" % llm.__class__.__dict__)
# 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
# 指定设备
# llm.set_device("cuda:3")


outputs = llm.generate(prompts, sampling_params)  # Generate texts from the prompts.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# sum_time = 0
# for i in range(10):
#     # 记录当前时间
#     start = time.time()
#     outputs = llm.generate(prompts, sampling_params)  # Generate texts from the prompts.
#     end = time.time()
#     sum_time += end - start
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# print("Average time: %f s" % (sum_time / 10))
