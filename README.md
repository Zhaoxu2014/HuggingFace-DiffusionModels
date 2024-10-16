# HuggingFace-DiffusionModels

##[HuggingFace Diffusion Models Class](https://github.com/darcula1993/diffusion-models-class-CN/)  

###Unit2: Fine-Tuning, Guidance and Conditioning  

Hands-On Notebook: 1. [Fine-tuning and Guidance](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)  


Fine-Tuning: 微调：  
Consideration 2: Our loss signal is extremely noisy, since we're only working with four examples at random noise levels for each step. This is not ideal for training. One fix is to use an extremely low learning rate to limit the size of the update each step. It would be even better if we could find some way to get the same benefit we would get from using a larger batch size without the memory requirements skyrocketing...  
Enter [gradient accumulation](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html#:~:text=Simply%20speaking%2C%20gradient%20accumulation%20means,might%20find%20this%20tutorial%20useful.). If we call `loss.backward()` multiple times before running `optimizer.step()` and `optimizer.zero_grad()`, then PyTorch accumulates (sums) the gradients, effectively merging the signal from several batches to give a single (better) estimate which is then used to update the parameters. This results in fewer total updates being made, just like we'd see if we used a larger batch size. This is something many frameworks will handle for you (for example, [🤗 Accelerate makes this easy](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation) but it is nice to see it implemented from scratch since this is a useful technique for dealing with training under GPU memory constraints! As you can see from the code above (after the `# Gradient accumulation` comment) there really isn't much code needed.  
考虑因素2： 我们的损失值曲线简直像噪声一样混乱！这是因为每一次迭代我们都只用了四个训练样本，而且加到它们上面的噪声水平还都是随机挑选的。这对于训练来讲并不理想。一种弥补的措施是，我们使用一个非常小的学习率，限制每次更新的幅度。但我们还有一个更好的方法，既能得到和使用更大的 batch size 一样的收益，又不需要让我们的内存爆掉。  
点击这里看看：[gradient accumulation](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html#:~:text=Simply%20speaking%2C%20gradient%20accumulation%20means,might%20find%20this%20tutorial%20useful.)。如果我们多运行几次`loss.backward()`后再调用`optimizer.step()`和`optimizer.zero_grad()`，PyTorch 就会把梯度累积（加和）起来，这样多个批次的数据产生的更新信号就会被高效地融合在一起，产出一个单独的（更好的）梯度估计用于参数更新。这样做会减少参数更新的总次数，就正如我们使用更大的 batch size 时希望看到的一样。梯度累积是一个很多框架都会替你做的事情（比如这里：[🤗 Accelerate makes this easy](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)），但这里我们从头实现一遍也挺好的，因为这对你在 GPU 内存受限时训练模型非常有帮助。正如你在上面代码中看到的那样（在注释 `# Gradient accumulation` 后），其实也不需要你写很多代码。  

解决方法：使用通义千问帮我按要求修改代码。  
```Python
num_epochs = 2  # @param
lr = 1e-5  # 2param
grad_accumulation_steps = 2  # @param

optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)

losses = []

for epoch in range(num_epochs):
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            image_pipe.scheduler.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

        # Get the model prediction for the noise
        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

        # Compare the prediction with the actual noise:
        loss = F.mse_loss(
            noise_pred, noise
        )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

        # Store for later plotting
        losses.append(loss.item())

        # Update the model parameters with the optimizer based on this loss
        loss.backward(loss)

        # Gradient accumulation:
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(
        f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):])/len(train_dataloader)}"
    )

# Plot the loss curve:
plt.plot(losses)
```
```
提问：
我正在进行对预训练过的扩散模型进行微调，以上是我的训练循环部分的代码。代码运行后，损失值曲线简直像噪声一样混乱。这是因为每一次迭代我们都只用了四个训练样本，而且加到它们上面的噪声水平还都是随机挑选的。这对于训练来讲并不理想。如果我们多运行几次loss.backward()后再调用optimizer.step()和optimizer.zero_grad()，PyTorch 就会把梯度累积（加和）起来，这样多个批次的数据产生的更新信号就会被高效地融合在一起，产出一个单独的（更好的）梯度估计用于参数更新。  

请根据以上文字描述，帮我修改代码。
```
```
回答：
为了改进你的训练循环，使得梯度累积可以更有效地工作，我们可以确保在每个epoch中对整个数据集进行完整的遍历，并且在累积了grad_accumulation_steps个批次的梯度之后才更新模型参数。这样可以帮助平滑损失曲线，因为每次参数更新都会基于更多的样本。

这里是修改后的代码段：
```
```Python
# 假设image_pipe, train_dataloader, device, F等变量已经定义好
num_epochs = 2  # @param
lr = 1e-5  # @param
grad_accumulation_steps = 2  # @param

# 初始化优化器
optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)

losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0  # 用于计算每个epoch的平均损失
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        clean_images = batch["images"].to(device)
        
        # 添加噪声到图片
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0,
            image_pipe.scheduler.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
        
        # 预测噪声
        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        epoch_loss += loss.item()  # 累积损失
        
        # 反向传播，但不立即执行梯度下降
        loss.backward()

        # 如果累积了足够的梯度，则更新权重并清空梯度
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # 对于最后一个batch，如果它没有被累积到grad_accumulation_steps次，也需要更新权重
    if (step + 1) % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # 记录每个epoch的平均损失
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch} average loss: {avg_epoch_loss}")

# 绘制损失曲线
plt.plot(losses)
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
```
```
这段代码中的主要变化包括：

在每个epoch结束时检查是否需要执行额外的optimizer.step()和optimizer.zero_grad()调用，以处理那些没有达到grad_accumulation_steps步数就被epoch结束打断的情况。
epoch_loss用来存储当前epoch的所有损失之和，然后在每个epoch结束时计算平均损失并打印出来。
losses列表现在记录的是每个epoch的平均损失，而不是每一步的损失，这有助于绘制更加平滑的损失曲线。
```
