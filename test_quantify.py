from transformers import AutoModelForCausalLM, AutoTokenizer
from quanti_fp4 import FP4SymmetricConfig
# from quanti import Int8SymmetricConfig
import torch  # 导入torch
# quant_config = FP4SymmetricConfig(
#     modules_to_not_convert=["lm_head"]  # 保持输出层不量化
# )




quant_config = FP4SymmetricConfig(
    modules_to_not_convert=["lm_head"]  # 保持输出层不量化
)
# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
   "microsoft/phi-2",
    quantization_config=quant_config,
    # force_download=True,
    device_map="auto",               # 自动分配GPU/CPU
    trust_remote_code=True,          # Phi-2需要此参数 
    torch_dtype=torch.float16,
)




# quant_config = {
#     "load_in_4bit": True,
#     "bnb_4bit_quant_type": "fp4",  # 或 "fp4"
#     "bnb_4bit_compute_dtype": torch.float16,
#     "bnb_4bit_use_double_quant": False,  # 二次量化（进一步压缩）
# }

# # 加载量化模型
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/phi-2",
#     quantization_config=quant_config,
#     device_map="auto",  # 自动分配 GPU/CPU
#     trust_remote_code=True,  # Phi-2 需要此参数
# )
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True  # 必须与模型一致
)

# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}, dtype={param.dtype}")
#     print(param.flatten()[:10])  # 只看前10个元素，避免爆屏
#     print("-" * 80)



# # 遍历已加载模型的参数
# for name, param in model.named_parameters():
#     if 'weight' in name or 'bias' in name:
#         # 打印参数的dtype，查看是否为 int8
#         print(f"{name}: {param.dtype}")
#         if param.dtype == torch.int8:
#             print(f"{name} has been quantized to INT8!")
#         else:
#             print(f"{name} is not quantized.")
#     # 只检查前几个层，防止输出过多
#     if 'q_proj' in name or 'k_proj' in name:
#         break


# for name, buffer in model.named_buffers():
#     if 'weight' in name or 'bias' in name:
#         print(f"{name}: {buffer.dtype}")
#         if buffer.dtype == torch.int8:
#             print(f"{name} has been quantized to INT8 (buffer)!")
#         else:
#             print(f"{name} is not quantized (buffer).")


for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.shape}, dtype={buffer.dtype}")
    print(buffer.flatten()[:10])  # 只看前10个元素，避免爆屏
    print("-" * 80)
    


# # 单独提取所有 buffer
# buffers_only = {name: buffer for name, buffer in model.named_buffers()}

# # 保存 buffer
# torch.save(buffers_only, "quantized_buffers_only_Int8SymmetricConfig.pth")

# def print_model_layers(model):
#     for name, module in model.named_modules():
#         if hasattr(module, 'weight'):
#             print(f"{name}: {module.weight.shape}, stored_as={module.weight.dtype}")
# print_model_layers(model)
# 打印第一个线性层的类型
# for name, module in model.named_modules():
#         print(f"层 {name} 的类型是: {type(module)}")