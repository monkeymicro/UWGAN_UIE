# underwater image process method and paper

传统方法和基于深度学习的方法总结：

https://github.com/CXH-Research/Underwater-Image-Enhancement

# GAN部分程序说明

这份基于 PyTorch 版本的 UWGAN（Underwater GAN）代码，其核心思想是**“物理模型驱动与数据驱动的结合”**。它并没有让生成器（Generator）像传统的黑盒网络那样自由发挥，而是将**经典的物理成像公式直接硬编码到了生成器的网络结构中**。

下面我为你梳理这份代码的程序设计说明与核心原理，分为物理公式处理、参数学习约束以及完整的训练流程。

---

### 一、 物理公式的处理方法：水下成像模型 (Jaffe-McGlamery)

生成器（`WCGenerator`）的前向传播严格遵循了简化的水下光学成像模型。一幅水下图像的最终像素强度通常由三部分组成：



#### 1. 直接衰减 (Direct Attenuation)
光在水中传播时会被吸收和散射，导致能量随距离指数级衰减。不同波长的光（红、绿、蓝）衰减率不同。
在代码中，这被建模为：
$$I_{direct}(x) = J(x) \cdot e^{-\eta_c \cdot d(x)}$$
* $J(x)$：输入的清晰空气图像 (`clear_image`)。
* $d(x)$：对应的深度图 (`depth`)。
* $\eta_c$：不同通道的衰减系数，即代码中的 `eta_r`, `eta_g`, `eta_b`。

#### 2. 后向散射 (Backscatter)
水中的悬浮颗粒会将光源（如太阳光或相机闪光灯）散射到相机镜头中，形成一层“雾蒙蒙”的背景光。
代码中没有使用严格的解析解，而是巧妙地用网络去**模拟这种分布**：
* 引入随机噪声 $z$，通过全连接层映射到图像空间。
* 结合深度图 $d(x)$：深度越大，后向散射通常越强。
* 通过独立的三通道卷积（`conv_r`, `conv_g`, `conv_b`）模拟不同波长散射的空间非线性分布，得到 $B_c(x)$。

#### 3. 雾霭效应 (Haze Effect) / 环境光
这是为了增加真实感额外引入的一项，模拟远处的环境光散射：
$$H_c(x) = (255 \cdot A) \cdot \left(1 - e^{-\eta_{haze} \cdot d(x)}\right) \cdot e^{-\eta_c \cdot d(x)}$$
* $A$：环境光强度的全局参数。
* $\eta_{haze}$：固定的雾霭衰减系数基数。

#### 4. 图像融合与全局增益
最后，将上述所有成分相加，并乘以一个全局透射率/增益参数 $B$：
$$I_{out}(x) = \left( I_{direct}(x) + B_c(x) + H_c(x) \right) \cdot B$$

---

### 二、 GAN 学习物理参数与约束实现

在这套架构中，生成器并不是在学习如何“画”出水下图像，而是**在学习水下环境的物理参数**。

#### 1. 参数的可学习化 (`nn.Parameter`)
代码中，衰减系数和全局参数都被定义为可学习的张量，并赋予了基于经验的初始值：
```python
self.eta_r = nn.Parameter(torch.normal(0.35, 0.01, size=(1, 1, 1, 1)))
self.eta_g = nn.Parameter(torch.normal(0.015, 0.01, size=(1, 1, 1, 1)))
# ...
```
在反向传播时，判别器传回的梯度会直接更新这些物理参数，而不是更新一个庞大的卷积核权重矩阵。

#### 2. 物理约束下的损失函数 (Physical Constraints)
物理世界中，衰减系数 $\eta$、光强 $A$ 和增益 $B$ **绝对不能是负数**。为了防止神经网络在寻找最优解时将这些参数更新为负值，代码在 Generator 的损失计算中加入了**单边惩罚项（正则化）**：
$$\mathcal{L}_{phys} = \lambda \sum_{p \in P} \max(0, -p)$$
*代码实现：*
```python
eta_r_loss = -torch.clamp(eta_r.min(), max=0.0) * 10000
```
**原理：** 当 `eta_r` 大于等于 0 时，`clamp` 的结果是 0，该项 loss 为 0；一旦 `eta_r` 变成负数（比如 -0.05），该项 loss 就会变成 500 (乘以一万倍的惩罚)，迫使优化器在下一步将参数拉回正数区间。

**注意**：
使用softplus函数处理所有动态参数，直接规避物理量小于零没有意义的问题，进行了变换后计算。实际训练效果确实比使用惩罚项loss效果更好。因此不再使用惩罚项的方式，实际代码中删除了乘法代码。


### 三、 完整的训练流程 (Training Flow)

这个架构是一个标准的无监督/半监督 Image-to-Image 转换流程，不需要成对的（Paired）空气-水下图像。



[Image of Generative Adversarial Network architecture diagram](imgs/licensed-image.jpg)


**Step 1: 数据准备 (Data Loading)**
* 随机抓取真实的水下图像（$I_{real\_water}$）。
* 随机抓取清晰的空气图像（$I_{air}$）及其对应的深度图（$D_{air}$）。

**Step 2: 判别器更新 (Update D)**
* **真实样本：** 将 $I_{real\_water}$ 送入判别器 $D$，计算 BCE Loss，期望输出为 1。
* **生成样本：** 生成器 $G$ 接收 $I_{air}$、$D_{air}$ 和随机噪声 $z$，利用当前的物理参数 ($\eta_r, \eta_g, \eta_b, A, B$) 计算出合成的水下图像 $I_{fake\_water}$。
* 将 $I_{fake\_water}$ 送入判别器 $D$（阻断梯度传向 G），期望输出为 0。
* 合并 Loss，更新 $D$ 的参数。

**Step 3: 生成器更新 (Update G)**
* 生成器 $G$ 再次生成 $I_{fake\_water}$。
* 将其送入判别器 $D$，此时期望 $D$ 的输出为 1（欺骗判别器），计算对抗损失 $\mathcal{L}_{GAN}$。
<!-- * 计算物理约束损失 $\mathcal{L}_{phys}$。 -->
* 总损失 $\mathcal{L}_G = \mathcal{L}_{GAN}$。
* 反向传播，此时不仅更新模拟后向散射的卷积层（`conv_r`, `conv_g` 等），**更核心的是更新那些物理参数** ($\eta_r, \dots, B$)。
* *(注：代码中 G 连续更新两次，是为了防止 D 过于强大导致 G 的梯度消失，这是早期 GAN 训练的一种常见技巧。)*

### 总结
这种设计的精妙之处在于：它将一个高度复杂的图像生成任务，**降维**成了一个**物理参数估计**任务。GAN 不需要去猜测水波纹或偏色的底层逻辑，它只需要通过对抗学习，找到一组最符合当前真实水下数据集分布的 $\eta_r, \eta_g, \eta_b$ 即可。这就保证了生成的图像在物理上是绝对合理且可解释的。

你说得很对！“让 G 连续更新两次”是一种非常早期的、偏经验主义的启发式技巧（Hack）。在早期的 DCGAN 时代，因为判别器（D）太容易把生成器（G）按在地上摩擦，导致 $G$ 的梯度变成 0（梯度消失），研究人员只能用这种“给 G 走后门”的笨办法。

但这种方法治标不治本，现在的学术界和工业界早就有了更优雅、从**数学和网络结构根源**上解决问题的现代训练方法。目前主流且更强大的 GAN 训练技术主要集中在以下四个方向：

---

### 1. 从“损失函数”开刀：告别梯度消失
早期 GAN 容易崩溃是因为使用了交叉熵损失（Sigmoid Cross Entropy）。当 D 很强时，假图被判为 0 的概率极高，此时交叉熵函数的梯度几乎为 0，$G$ 就“学不动”了。

* **WGAN-GP (Wasserstein GAN with Gradient Penalty)：** 这是 GAN 历史上的一大里程碑。它抛弃了“判断真假”的二分类思路，转而计算真实分布和生成分布之间的“推土机距离”（Earth Mover's Distance）。判别器不再叫 Discriminator，而是叫 Critic（打分器）。**优点是：即使 D 训练得极其完美，$G$ 依然能获得稳定、线性的梯度，彻底告别梯度消失。** 
* **Hinge Loss 或 LSGAN (Least Squares GAN)：** 用均方误差（MSE）或 Hinge 损失代替交叉熵。这也是目前（比如 BigGAN、SAGAN 中）非常流行的做法，计算更简单，且能在边缘区域继续提供梯度。

### 2. 给判别器“戴上紧箍咒”：权重归一化
与其让 $G$ 偷跑，不如限制 $D$ 的能力，让它不要过于“尖锐”和自信。

* **谱归一化 (Spectral Normalization, SN)：** **这是目前极其推荐的方法，可以说是现代 GAN 的标配！** 它通过限制判别器每一层权重矩阵的最大奇异值，强行让判别器满足“利普希茨连续性”（Lipschitz continuity）。简单来说，就是让判别器的打分曲线变得非常平滑。你只需要在 $D$ 的每一个卷积层上套一个 `SpectralNorm`，训练就会变得异常稳定，根本不需要让 $G$ 跑两次。
* **R1 正则化 (R1 Regularization)：** 这是 StyleGAN 系列爱用的绝招。它专门针对真实图片（Real Images）计算判别器的梯度惩罚，迫使判别器在真实数据流形附近的决策边界保持平滑。

### 3. 更聪明的优化策略：TTUR
* **双时间尺度更新规则 (Two Time-Scale Update Rule, TTUR)：** 这篇论文从数学上证明了，与其改变更新**次数**（G 跑两次，D 跑一次），不如改变**学习率（Learning Rate）**。
    * 现代 GAN 训练通常会让 $D$ 的学习率比 $G$ **大**（例如 $lr_D = 0.0004$, $lr_G = 0.0001$）。
    * **原理：** 在使用了 WGAN 或谱归一化的前提下，让 $D$ 快速收敛到最优状态，这样它每次反馈给 $G$ 的梯度才是最准确、最有价值的。这与早期“怕 D 太强”的思路完全相反，现代 GAN 是“保证 D 不会梯度消失的前提下，让 D 越准越好”。

### 4. 架构层面的进化
* **渐进式增长 (Progressive Growing)：** 先训练 4x4 的超低分辨率，等 G 和 D 在低分辨率下打成平手了，再慢慢加层数，训练 8x8、16x16…… 这种方法（ProGAN首创）极大地降低了高分辨率图像生成的训练难度。
* **自注意力机制 (Self-Attention, SAGAN)：** GAN 以前只擅长生成局部的纹理（比如狗的毛发），但不擅长把握全局结构（比如狗应该有四条腿）。在 G 和 D 中引入自注意力层，能帮助网络理解图像的远距离依赖关系，极大提升生成质量。

---

### 总结：现在应该怎么做？

如果你现在要手写或优化一个 GAN 模型，**建议抛弃“G 更新两次”的代码，直接采用以下“现代标准套餐”**：

1.  判别器 $D$ 的所有卷积层使用 **谱归一化 (Spectral Normalization)**。
2.  损失函数换成 **Hinge Loss** 或 **WGAN-GP**。
3.  使用 **TTUR**，给 $D$ 分配一个稍大一点的学习率，给 $G$ 分配一个稍小一点的学习率。

这套组合拳几乎可以保证你的 GAN 平稳收敛，且生成的质量远超早期的传统架构。 当前的代码中，这三点都使用了。


### 1. 当前问题解答

**你目前写的确实是 LSGAN（最小二乘 GAN，使用的是 MSE 损失）。**

**损失函数的问题解决了吗？** **解决了一大半。** 将 Sigmoid 交叉熵换成 MSE（LSGAN）是一个巨大的进步。它能有效防止在判别器（D）太聪明时生成器（G）的梯度变为 0（梯度消失），从而拉着 G 强行往真实分布靠近。对于很多任务来说，LSGAN 的效果已经足够好了。

**但为什么 WGAN-GP 更好？**
LSGAN 依然是在试图拉近两者的均值，而 **WGAN-GP（带有梯度惩罚的 Wasserstein GAN）** 计算的是真实分布和生成分布之间的“推土机距离”（Earth Mover's Distance）。
它的理论非常完美：**无论 D 有多强，WGAN-GP 都能给 G 提供几乎完美的线性线性梯度，彻底杜绝模式崩溃（Mode Collapse），是目前 GAN 训练公认的“金标准”。**

---

### 2. 如何改为更好的 WGAN-GP？

要将你的代码升级为 WGAN-GP，需要修改损失函数的计算方式，并引入**梯度惩罚（Gradient Penalty）**。

请将你注释掉和刚刚写的那段 `self.d_loss_real`、`self.d_loss_fake` 和 `self.g_loss` 替换为以下代码：

```python
        # ==========================================
        # WGAN-GP (Wasserstein GAN with Gradient Penalty) 损失实现
        # ==========================================
        
        # 1. 基础 Wasserstein 损失 (注意：不使用任何 Sigmoid 或 Square，直接用 Logits)
        # D 试图让真图的分数更高(为正)，假图的分数更低(为负)
        self.d_loss_real = -tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        
        # 2. 梯度惩罚 (Gradient Penalty)
        # 获取当前实际的 batch_size (应对最后一个 batch 可能不满的情况)
        current_batch_size = tf.shape(self.water_inputs)[0]
        
        # 在真假图像之间进行随机线性插值
        alpha = tf.random_uniform(shape=[current_batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = alpha * self.water_inputs + (1. - alpha) * self.G
        
        # 将插值图像送入判别器获取输出 (复用 D 的权重)
        _, D_logits_interp, _ = self.discriminator(image=interpolates, reuse=True)
        
        # 计算 D_logits_interp 对插值图像 interpolates 的梯度
        gradients = tf.gradients(D_logits_interp, [interpolates])[0]
        
        # 计算梯度的 L2 范数 (加上 1e-8 防止 tf.sqrt 在 0 处求导出现 NaN)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]) + 1e-8)
        
        # 惩罚项：鼓励梯度的 L2 范数趋近于 1 (Lipschitz 连续性约束)
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
        
        # 3. 最终组合 Loss
        # 梯度惩罚系数 lambda 通常设为 10.0
        lam = 10.0
        self.d_loss = self.d_loss_real + self.d_loss_fake + lam * gradient_penalty
        
        # G 试图让判别器给假图打的分数越高越好
        self.g_loss = -tf.reduce_mean(self.D_logits_)
```

---

### 3. WGAN-GP 的两个关键“避坑”提醒

如果你使用了上面的 WGAN-GP 代码，你必须注意两件事，否则模型会直接崩掉：

#### ⚠️ 避坑一：判别器绝对不能使用 Batch Normalization (BN)
WGAN-GP 惩罚的是判别器对**单一独立样本**的梯度。而 Batch Normalization 会把整个 Batch 的数据混在一起计算均值和方差，这违背了 WGAN-GP 的数学前提。

**好消息是：** 你在之前的修改中，已经在 `ops.py` 里将 `batch_norm` 替换为了 `slim.instance_norm`（实例归一化）。**Instance Normalization 和 WGAN-GP 是完美兼容的！** 所以你的判别器网络结构不需要做任何改动，直接用就行。

#### ⚠️ 避坑二：优化器 (Optimizer) 的动量参数
WGAN 论文中明确指出，使用带有动量（Momentum）的优化器（如 Adam 的默认参数）会导致训练不稳定。

在你的 `train` 函数中，需要**去掉动量，或者把 Adam 的 $\beta_1$ 设为 0**（有些论文用 0.5 也行，但 0 是最稳的）。

在 `train` 函数里修改你的优化器：
```python
        # WGAN-GP 推荐的优化器参数：不要使用动量 (beta1=0.0, beta2=0.9)
        d_optim = tf.train.AdamOptimizer(d_lr, beta1=0.0, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_vars)
        
        g_optim = tf.train.AdamOptimizer(g_lr, beta1=0.0, beta2=0.9) \
            .minimize(self.g_loss, var_list=self.g_vars)
```

**总结：** 换上 WGAN-GP 后，你会发现 $d\_loss$ 和 $g\_loss$ 的数值不再具有直观的“准确率”含义（它们可以变成负数，这是正常的），因为它们代表的是距离分布。只要两者的曲线不再呈现“一飞冲天”的爆炸趋势，你的模型就是在健康地学习。