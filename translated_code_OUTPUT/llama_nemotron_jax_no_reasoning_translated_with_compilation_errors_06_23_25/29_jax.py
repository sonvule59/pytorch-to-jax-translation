18.7


***

JAX Code

jax
import jax
import jax.nn as jnn
import jax.nn.functional as jfu
import jax.random as jrandom

class Bottleneck(jax.nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super().__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = jnn.conv2d(last_planes, in_planes, kernel_size=1, stride=1, padding=0, init_mode='zeros')
        self.bn1 = jnn.BatchNorm2d(in_planes)
        self.conv2 = jnn.conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=floor(stride / 2), groups=32, init_mode='zeros')
        self.bn2 = jnn.BatchNorm2d(in_planes)
        self.conv3 = jnn.conv2d(in_planes, out_planes + dense_depth, kernel_size=1, stride=1, padding=0, init_mode='zeros')
        self.bn3 = jnn.BatchNorm2d(out_planes + dense_depth)

        self.shortcut = jax.nn.Sequential()
        if first_layer:
            self.shortcut = jax.nn.Sequential(
                jnn.conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, padding=0, init_mode='zeros'),
                jnn.BatchNorm2d(out_planes + dense_depth)
            )

    def forward(self, x):
        out = jfu.relu(self.bn1(self.conv1(x)))
        out = jfu.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = jax.cat([x[:, :d, :, :]+out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], axis=1)
        out = jfu.relu(out)
        return out

class DPN(jax.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = jnn.conv2d(3, 64, kernel_size=3, stride=1, padding=1, init_mode='zeros')
        self.bn1 = jnn.BatchNorm2d(64)
        self.last_planes = 64
        def make_layer(in_planes, out_planes, num_blocks, dense_depth, strides):
            strides = [strides[0]] + [1]*num_blocks + [strides[1]]*(num_blocks-1)
            layers = []
            for i, stride in enumerate(strides):
                layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
                self.last_planes = out_planes + (i+2) * dense_depth
            return jax.nn.Sequential(*layers)
        
        self.layer1 = make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], [1,2])
        self.layer2 = make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], [2,2])
        self.layer3 = make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], [2,2])
        self.layer4 = make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], [2,2])
        self.linear = jnn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 10)

    def forward(self, x):
        out = jfu.relu(self.bn1(self.conv1(x)))
        out = self.layer1(output)
        out = self.layer2(output)
        out = self.layer3(output)
        out = self.layer4(output)
        out = jfu.avg_pool2d(out, 4)
        out = out.reshape((*out.shape, out.size(1)))
        return self.linear(out)

def DPN26():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_planes': (96, 192, 384, 768),