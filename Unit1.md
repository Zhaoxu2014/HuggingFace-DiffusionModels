Hands-On Notebook: 1. [Introduction to Diffusers](https://colab.research.google.com/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/01_introduction_to_diffusers_CN.ipynb)  

<br><br>

**关键流程：** <br>

1.安装必要的库：安装 transformers, diffusers 以及其他相关的库。
```
pip install transformers diffusers datasets
```
2.导入所需的模块：从diffusers和其他相关库中导入所需的类和函数。
``` Python
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset
import torchvision
from torchvision import transforms
# ......
```
3.准备数据集：
``` Python
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")  # 使用来自 Hugging Face Hub 的 “smithsonian_butterflies_subset” 数据集
# dataset = load_dataset("imagefolder", data_dir="path/to/folder")  # 或从本地文件夹加载数据集

image_size = 32  # 训练的图像尺寸为32×32px

batch_size = 64  # 一次训练所选取的样本数为64

# 数据预处理
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

# 从数据集创建数据加载器以批量提供转换后的图像
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)
```
4.定义和配置模型：创建或加载一个扩散模型，并对其进行适当的配置。
``` Python
model = UNet2DModel(
    sample_size=32,  # 图片大小
    in_channels=3,   # 输入图片的通道数
    out_channels=3,  # 输出图片的通道数
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=(
        "DownBlock2D",  # 常规 ResNet 下采样块
        "DownBlock2D",
        "AttnDownBlock2D",  # 具有空间自注意力的 ResNet 下采样块
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # 具有空间自注意力的 ResNet 上采样块
        "UpBlock2D",
        "UpBlock2D",  # 常规 ResNet 上采样块
    ),
)
```
5.训练模型：使用数据集对模型进行训练，包括定义训练循环、优化器、损失函数等。
``` Python
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")  # 调度器，训练过程中总的时间步数为1000，噪声强度的调度方式为平方余弦调度方式。

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)  # 优化器，使用AdamW优化器，学习率设置为0.0004。

losses = []  # 保存训练过程的loss值。

for epoch in range(30):  # 总共训练30个周期（epochs）。
    for step, batch in enumerate(train_dataloader):  # 遍历每个批次的数据。

        # 准备数据
        clean_images = batch["images"].to(device)  # 将干净图像移动到CPU或GPU。

        # 生成噪声
        noise = torch.randn(clean_images.shape).to(clean_images.device)  # 生成与干净图像相同形状的随机噪声。
        bs = clean_images.shape[0]

        # 随机选择时间步
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()  # 为每个图像随机选择一个时间步，范围是从0到num_train_timesteps-1。

        # 添加噪声
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)  # 根据选定的时间步和噪声强度，将噪声添加到干净图像中，生成带噪声的图像。

        # 将带噪声的图像和时间步输入模型，得到模型对噪声的预测。
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # 计算loss
        loss = F.mse_loss(noise_pred, noise)  # 计算预测噪声和真实噪声之间的均方误差损失。
        loss.backward(loss)  # 反向传播损失，计算梯度。
        losses.append(loss.item())  # 将损失值记录到losses列表中。

        # 更新模型参数
        optimizer.step()  # 使用优化器更新模型参数。
        optimizer.zero_grad()  # 清除梯度，准备下一个批次的训练。

    # 打印每5个周期的平均损失
    if (epoch + 1) % 5 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
```
6.生成图像
``` Python
# 方法1：建立一个Pipeline

# 建立pipeline并输出图像
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
pipeline_output = image_pipe()
pipeline_output.images[0]

# 将此pipeline保存到本地，是上传模型至Huggingface Hub的关键
image_pipe.save_pretrained("my_pipeline")  # 保存后"my_pipeline"目录下有model_index.json  scheduler  unet这样的文件。
```
7.将模型Push到Huggingface Hub
``` Python
# 登陆Huggingface
huggingface-cli login

from huggingface_hub import get_full_repo_name

model_name = "sd-class-butterflies-32"
hub_model_id = get_full_repo_name(model_name)

# HfApi 是Hugging Face Hub的API接口，用于执行各种操作，如上传文件和文件夹。
# create_repo 函数用于在Hugging Face Hub上创建一个新的仓库。
from huggingface_hub import HfApi, create_repo  

create_repo(hub_model_id)  # 在Hugging Face Hub上创建一个新的仓库。

#将pipeline保存至本地生成的文件上传到Hugging Face Hub上的指定仓库。
api = HfApi()
api.upload_folder(
    folder_path="my_pipeline/scheduler",  # 指定路径，上传scheduler。
    path_in_repo="",  # 文件夹将保存在远程仓库中的路径，如为空字符串""，则文件夹将直接上传到仓库的根目录。
    repo_id=hub_model_id
)
api.upload_folder(
    folder_path="my_pipeline/unet",  # 指定路径，上传unet。
    path_in_repo="",
    repo_id=hub_model_id)
api.upload_file(
    path_or_fileobj="my_pipeline/model_index.json",  # 指定路径，上传model_index.json。
    path_in_repo="model_index.json",
    repo_id=hub_model_id,
)
```
8.使用该模型
``` Python
# 使用DDPMPipeline的from_pretrained ()方法加载此预训练模型来生成图像。
from diffusers import DDPMPipeline

image_pipe = DDPMPipeline.from_pretrained(hub_model_id)
pipeline_output = image_pipe()
pipeline_output.images[0]
```



















