import jax.numpy as jnp
import jax.data.augment
import jax.data.preproc
import jax.numpy.ops

class MscocoJAX(data.Dataset):
    def __init__(self, is_train=True, **kwargs):
        super().__init__(**kwargs)
        self.sigma = kwargs['sigma']
        self.label_type = kwargs['label_type']
        self.year = kwargs['year']
        self.jsonfile = kwargs['jsonfile']
        self.img_folder = kwargs.get('image_path', '')
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = './data/coco/mean.pth.tar'
        meanstd = torch.load(meanstd_file) if os.path.isfile(meanstd_file) else {
           'mean': jnp.zeros(3),
           'std': jnp.zeros(3)
        }
        if self.is_train:
            annotations = self.anno[self.train]
            total = jnp.zeros(3)
            total_sum = jnp.zeros(3)
            count = 0
            for ann in annotations:
                img_path = os.path.join(self.img_folder, ann['img_paths'])
                img = load_image(img_path)
                mean = img.view(-1).mean(1)
                std = img.view(-1).std(1)
                total += mean
                total_sum += std
                count += 1
            meanstd = {
               'mean': total / count,
               'std': total_sum / count
            }
        return meanstd

    def __getitem__(self, index):
        a = self.anno[self.valid[index]] if not self.is_train else self.anno[self.train[index]]
        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = jnp.array(a['joint_self'], dtype=jnp.float32)
        c = jnp.array(a['objpos'], dtype=jnp.float32) - 1
        s = jnp.array(a['scale_provided'], dtype=jnp.float32)

        if c[0]!= -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        nparts = pts.size(0)
        img = load_image(img_path)

        r = jnp.zeros(1)
        if self.is_train:
            r[0] = jnp.random.uniform(-2, 2, jnp.array([1], dtype=jnp.float32))
            s = s * jnp.random.uniform(1-sf, 1+sf, jnp.array([1], dtype=jnp.float32)).add_(1).clamp(1-sf, 1+sf)[0]
            if jnp.random.random() <= 0.5:
                img = flip_image(img)
                pts = shuffle_lr(pts, width=img.shape[2], jnp.array([1], dtype=jnp.float32))
                c[0] = img.shape[2] - c[0]

        inp = crop(img, c, s, [self.inp_res, self.inp_res], r)
        inp = color_normalize(inp, self.mean, self.std)

        tpts = pts.clone()
        target = jnp.empty((nparts, self.out_res, self.out_res))
        target_weight = pts[:, 2].view(nparts, 1)
        for i in range(nparts):
            if pts[i, 2] > 0:
                transformed_pts = transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], r)
                target[i] = draw_labelmap(target[i], pts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] = jnp.array([vis], dtype=jnp.float32)

        meta = {'index': index, 'center': c,'scale': s, 'pts': pts, 'target_weight': target_weight['target_weight']}

        return (inp, target, meta), (target_weight,)

    def __len__(self):
        return len(self.valid) if not self.is_train else len(self.train)

def jax_coco(**kwargs):
    return MscocoJAX(**kwargs)