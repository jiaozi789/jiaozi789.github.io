---
title: "Transformers实战05-模型量化"
date: 2025-09-18T16:55:17+08:00
# bookComments: false
# bookSearchExclude: false
---


# 简介
模型量化（Model Quantization）是一种优化技术，旨在减少机器学习模型的计算资源需求和存储空间，同时在精度损失最小化的前提下提高推理效率。量化通过将模型权重和激活函数的数值从高精度（如 32 位浮点数）转换为低精度（如 8 位整数），显著减少了模型大小和计算复杂度。

## 主要类型

1. **静态量化（Post-Training Quantization, PTQ）**
    
    - 在模型训练完成后进行量化。
    - 通过分析训练数据的分布，将权重和激活函数映射到低精度表示。
    - 不需要重新训练模型。
    - 适用于对性能影响较小的场景。
2. **动态量化（Dynamic Quantization）**
    
    - 在推理时动态地将浮点数转换为低精度整数。
    - 在运行过程中对激活函数进行量化。
    - 比静态量化更简单，因为不需要分析训练数据。
    - 对推理速度有显著提升，尤其是对模型输入依赖较少的层（如全连接层）。
3. **量化感知训练（Quantization-Aware Training, QAT）**
    
    - 在训练过程中模拟量化影响。
    - 模型在训练过程中考虑量化误差，以便在量化后保持更高的精度。
    - 比静态量化和动态量化需要更多的计算资源，但精度损失最小。
    - 适用于对精度要求较高的应用。

这里例子就演示下动态量化，bitsandbytes本身以上三种都支持。
## 量化的优点
- **减小模型大小**：通过将权重和激活函数表示从 32 位浮点数转换为 8 位整数，模型大小可以显著减少。
- **加快推理速度**：低精度运算速度更快，可以显著提高推理效率。
- **降低内存带宽需求**：低精度表示占用更少的内存，减少了内存带宽的需求。
## 量化的缺点
- **精度损失**：由于数值表示的精度降低，模型可能会经历一定程度的精度损失，具体程度取决于模型结构和数据分布。
- **复杂性增加**：在某些情况下，量化过程可能会增加模型部署的复杂性，尤其是需要进行量化感知训练时。
## 量化过程
以下过程只是一种最简单的思路，方便理解，实际要比这更复杂。
### 量化过程

1. **确定值域：** 首先，确定要量化的数据的值域范围。例如，假设我们有一组数据的值域为 $[min,max]$。
    
2. **确定量化级别：** 确定量化的级别或分辨率，这决定了将值域划分成多少个区间。在4位整数的情况下，共有 $2^4=16$ 个可能的值。
    
3. **线性映射：** 将原始数据映射到4位整数的范围内。通常使用线性映射来实现，计算公式如下：
$$\text{quantized\_value} = \frac{\text{original\_value} - \text{min}}{\text{max} - \text{min}} \times (\text{number of levels} - 1)$$
4. 这里的 number of levels 是16（对应4位整数的值域范围）。
### 反量化过程
解码反量化： 在使用量化数据进行计算之前，需要将其解码回原始的数据表示形式（如32位浮点数或其他高精度表示）。解码公式通常为：
$$\text{original\_value} = \text{quantized\_value} \times \frac{\text{max} - \text{min}}{\text{number of levels} - 1} + \text{min}$$
这里的 quantized_value是是量化后的4位整数值,min和max是原始数据的最小值和最大值。

两个不同的原始值在量化后可能相同，被还原为同一个值。这种情况表明精度损失是不可避免的。为了减少这种精度损失带来的影响，通常采取以下策略：

1. **增加量化级别：** 增加量化级别（如使用8位、16位量化）以减少不同原始值被量化为同一个值的概率。
    
2. **量化感知训练（Quantization-aware training）：** 在训练过程中模拟量化误差，以提高模型在量化后的精度表现。
    
3. **非线性量化：** 使用对数量化或其他非线性量化方法，使得量化更适应数据的分布特性，从而减少精度损失。
    
4. **精细调节量化参数：** 通过精细调整量化的最小值、最大值和比例因子，尽量减少量化误差对关键值的影响。

## 精度和参数
模型中每个参数常见的存储类型包括：

- **FP32（32-bit Floating Point）**: 每个参数占用 4 字节（32 位），单精度浮点数（32位浮点数），范围大约：$[-3.4 \times 10^{38}, 3.4 \times 10^{38}]$。
- **FP16（16-bit Floating Point）**: 每个参数占用 2 字节（16 位），半精度浮点数使用16位（1位符号、5位指数、10位尾数），FP16的数值范围大约是 [−65504,65504]，大约 3 位有效数字。
- **INT8（8-bit Integer）**: 每个参数占用 1 字节（8 位），将模型的权重和激活值量化为8位整数（范围通常是0到255），相对于32位浮点数，精度的损失较小。8-bit量化比4-bit提供更好的精度，并且通常可以更接近原始模型的性能。
- **INT4（4-bit Integer）**: 每个参数占用4位，将模型的权重和激活值量化为4位整数（范围通常是-8到7或者0到15），因此相对于32位浮点数，它的精度显著降低。这种量化可以显著减小模型的大小和计算需求，但可能会损失一定的模型精度。

如何获取某个模型的精度了

```
import torch
from transformers import AutoModel, BertTokenizer
model_name="bert-base-chinese" #bert-base-uncased
model=AutoModel.from_pretrained(model_name)
#获取模型参数的精度
"""
    FP32（32-bit Floating Point）: 每个参数占用 4 字节（32 位）。
    FP16（16-bit Floating Point）: 每个参数占用 2 字节（16 位）。
    INT8（8-bit Integer）: 每个参数占用 1 字节（8 位）。
"""
dtype=list(model.parameters())[0].dtype
print("精度:",dtype)
total_params = sum(p.numel() for p in model.parameters())
dtype_to_bytes = {
    torch.float32: 4,  # FP32: 4字节
    torch.float16: 2,  # FP16: 2字节
    torch.int8: 1,     # INT8: 1字节
    torch.int32: 4,    # INT32: 4字节
    torch.int64: 8,    # INT64: 8字节
    torch.float64: 8,  # FP64 (double): 8字节
}
model_size = total_params * dtype_to_bytes[dtype]
print(f'Model size: {model_size / (1024**2):.2f} MB')
```
输出
```
精度: torch.float32
Model size: 390.12 MB
```
# 量化实例
## bitsandbytes
bitsandbytes 通过 PyTorch 的 k 位量化技术使大型语言模型的访问变得可行。bitsandbytes 提供了三个主要功能以显著降低推理和训练时的内存消耗：

- 8 位优化器采用区块式量化技术，在极小的内存成本下维持 32 位的表现。
- LLM.Int() 或 8 位量化使大型语言模型推理只需一半的内存需求，并且不会有任何性能下降。该方法基于向量式的量化技术将大部分特性量化到 8 位，并且用 16 位矩阵乘法单独处理异常值。
- QLoRA 或 4 位量化使大型语言模型训练成为可能，它结合了几种节省内存的技术，同时又不牺牲性能。该方法将模型量化至 4 位，并插入一组可训练的低秩适应（LoRA）权重来允许训练。
### 安装bitsandbytes
bitsandbytes 仅支持 CUDA 版本 11.0 - 12.5 的 CUDA GPU。
```
!pip install -U bitsandbytes
!pip install transformers
!pip install accelerate
```

### 4bit量化(加载)
加载并量化一个模型到4位，并使用bfloat16数据类型进行计算：
>您使用 bnb_4bit_compute_dtype=torch.bfloat16，这意味着计算过程中会反量化使用 bfloat16 数据类型，而存储时则可能使用4位表示。这解释了为什么您看到的 dtype 仍然是 fp16 或者 bfloat16。

BigScience 是一个全球性的开源AI研究合作项目，旨在推动大型语言模型（LLM）的发展。bloom-1b7 是 BigScience 项目下的一部分，具体来说，是一个包含约17亿参数的语言模型。

```
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
model_name="bigscience/bloom-1b7" 
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
)
dtype=list(model.parameters())[0].dtype
print("原始精度:",dtype)
dest_dtype=list(model_4bit.parameters())[0].dtype
print("量化精度:",dest_dtype)

# 检查模型的量化配置
print("量化配置:", model_4bit.config.quantization_config)

def print_model_info(model):
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    #print(f"Total parameters: {total_params / 1e6}M")
    return total_params

total_model_size=print_model_info(model)
total_model_4bit_size=print_model_info(model_4bit)
print("模型参数个数：",total_model_size)
print("量化后的模型参数个数：",total_model_4bit_size)

dtype_to_bytes = {
    torch.float32: 4,  # FP32: 4字节
    torch.float16: 2,  # FP16: 2字节
    torch.int8: 1,     # INT8: 1字节
    torch.int32: 4,    # INT32: 4字节
    torch.int64: 8,    # INT64: 8字节
    torch.float64: 8,  # FP64 (double): 8字节
}
model_size = total_model_size * dtype_to_bytes[dtype]
model_size = total_model_size * dtype_to_bytes[dtype]
print(f'origin Model size: {model_size / (1024**2):.2f} MB')
model_size = total_model_4bit_size * dtype_to_bytes[dest_dtype]
print(f'quan Model size: {model_size / (1024**2):.2f} MB')

model_4bit.save_pretrained("/tmp/p")
model.save_pretrained("/tmp/o")
```
输出：
```
原始精度: torch.float32
量化精度: torch.float16
量化配置: BitsAndBytesConfig {
  "_load_in_4bit": true,
  "_load_in_8bit": false,
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_quant_storage": "uint8",
  "bnb_4bit_quant_type": "fp4",
  "bnb_4bit_use_double_quant": false,
  "llm_int8_enable_fp32_cpu_offload": false,
  "llm_int8_has_fp16_weight": false,
  "llm_int8_skip_modules": null,
  "llm_int8_threshold": 6.0,
  "load_in_4bit": true,
  "load_in_8bit": false,
  "quant_method": "bitsandbytes"
}

模型参数信息： 1722408960
量化后的模型参数信息： 1118429184
origin Model size: 6570.47 MB
quan Model size: 2133.23 MB
```
总的参数个数减少。这通常是由于量化过程中进行了优化或者参数压缩的操作。
量化在深度学习中通常是指将模型中的浮点数参数转换为更低精度的整数或定点数表示，以节省内存和提高计算效率。

为啥量化模型的dtype是fp16了而不是int4，以下是对量化模型加载过程中 `dtype` 问题的一些解释：

1. **参数存储与计算类型的区别**：
    
    - 存储时，模型参数可能被压缩或量化为较低位宽的整数类型（如4位整数）。
    - 加载时，为了方便后续计算，这些参数可能会被解码为较高精度的浮点类型（如 `fp16` 或 `bfloat16`）。
2. **量化过程的具体实现**：
    
    - 许多量化库在加载模型时，会将低位宽的量化参数解码为浮点类型，以便在计算时可以直接使用这些参数。
    - 这就是为什么即使您使用了 `load_in_4bit=True`，在加载后检查参数的 `dtype` 时仍然看到的是 `fp16`。

通过查看模型保存的就可以确定了
查看量化的模型：
```
!ls /tmp/p -al --block-size=M | grep model
```
输出:
```
-rw-r--r-- 1 root root 1630M Aug  6 08:04 model.safetensors
```
可以看到我们之前在内存中打印的是2133.23（内存中计算还是会被反量化到bnb_4bit_compute_dtype指定类型，但是参数都是压缩后去掉了一些参数） ，存储后变成了1630M，比之前计算的少一些，说明存储使用了4bit。
在看下没有量化的模型：

```
!ls /tmp/o -al --block-size=M | grep model
```
输出了：
```
-rw-r--r-- 1 root root 4714M Aug  6 08:05 model-00001-of-00002.safetensors
-rw-r--r-- 1 root root 1857M Aug  6 08:05 model-00002-of-00002.safetensors
-rw-r--r-- 1 root root    1M Aug  6 08:05 model.safetensors.index.json
```
可以看到我们之前在内存中打印的是6570.47 MB ，存储后没变，分文件存储了4714M+1857M 。

###  8bit量化(加载)
代码和4bit相似，调整下配置即可

```
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```
同4bit代码，输出

```
原始精度: torch.float32
量化精度: torch.float16
量化配置: BitsAndBytesConfig {
  "_load_in_4bit": false,
  "_load_in_8bit": true,
  "bnb_4bit_compute_dtype": "float32",
  "bnb_4bit_quant_storage": "uint8",
  "bnb_4bit_quant_type": "fp4",
  "bnb_4bit_use_double_quant": false,
  "llm_int8_enable_fp32_cpu_offload": false,
  "llm_int8_has_fp16_weight": false,
  "llm_int8_skip_modules": null,
  "llm_int8_threshold": 6.0,
  "load_in_4bit": false,
  "load_in_8bit": true,
  "quant_method": "bitsandbytes"
}

模型参数信息： 1722408960
量化后的模型参数信息： 1722408960
origin Model size: 6570.47 MB
quan Model size: 3285.23 MB
```
可以看到8bit不需要指定内存计算的类型，量化内存计算精度默认就是fp16。
查看模型保存大小
```
!ls /tmp/p -al --block-size=M | grep model
#----------------------------------------------------------------------------------------------------
!ls /tmp/o -al --block-size=M | grep model
```
输出

```
-rw-r--r-- 1 root root 2135M Aug  6 08:30 model.safetensors
#----------------------------------------------------------------------------------------------------
-rw-r--r-- 1 root root 4714M Aug  6 08:30 model-00001-of-00002.safetensors
-rw-r--r-- 1 root root 1857M Aug  6 08:31 model-00002-of-00002.safetensors
-rw-r--r-- 1 root root    1M Aug  6 08:31 model.safetensors.index.json
```
### 验证效果
这里用之前的4bit模型来和原始模型比较

```
import time

def benchmark_model(model, input_text, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # 解码并打印生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "Hello, how are you?"
print("未量化模型性能测试：")
benchmark_model(model, input_text, tokenizer)
print("量化模型性能测试：")
benchmark_model(model_4bit, input_text, tokenizer)
```
输出

```
未量化模型性能测试：
Generated text: Hello, how are you? I hope you are doing well. I am a newbie in this
Inference time: 0.31 seconds
量化模型性能测试：
Generated text: Hello, how are you?"
"I'm fine," I said.
"I'm just a
Inference time: 0.62 seconds
```
这里看到量化的模型反而推理需要更多的时间，量化模型在理论上应该提高推理速度和减少内存占用,这里使用float16gpu显存占用肯定少了一半以上，但是推理速度比较慢，在实际应用中，可能会因为多个因素导致性能下降。
