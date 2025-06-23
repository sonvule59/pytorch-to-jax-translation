# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from jax import transformer
from jax.vmap import vmap
from jax.experimental import tap

# 定义 JAX 优化器
optimizer = jax.optim.SGD(vmap(net.parameters()))(lr=0.5)
# 定义损失函数
loss_func = jax.nn.mse(jnp.mean, loss_function=True)  # MSE loss

# 初始化设备
jax.tape()
x = jnp.array([jnp.linspace(-1, 1, 100)]).flatten(1)  # x data (array), shape=(100, )
y = x**2 + 0.2 * jnp.random.uniform(size=x.size())     # noisy y data (array), shape=(100, )

# 前向传播函数
@jax.transformer.compact
def forward(x):
    x_h_o = jnp.relu(net(x))  # 隐藏层激活函数
    y = net.predict(x_h_o)     # 输出层线性变换
    return y

# 训练循环
for t in range(100):
    prediction = forward(x)
    loss = loss_func(prediction, y)
    # 优化器更新
    optimizer.apply(jax.grad(loss, optimizer).apply)
    # 统计学习率
    if t % 5 == 0:
        # 画图逻辑
        import matplotlib.pyplot as plt
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.numpy(), prediction.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
    jax.vmap.tap(optimizer, loss)  # 保留学习率调节

# 清理环境
jax.clear()