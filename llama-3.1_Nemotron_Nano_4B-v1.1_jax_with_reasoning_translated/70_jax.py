import bisect
import torch
import jax.numpy as jnp
import jax
from mmcv import mmcv
from mmcv.runner import DistEvalHook as BaseDistEvalHook, EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm

def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    dynamic_milestones = [0]
    dynamic_milestones.extend([interval[0] for interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend([interval[1] for interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals

class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            dynamic_milestones, dynamic_intervals = _calc_dynamic_intervals(self.interval, dynamic_intervals)
            self.dynamic_milestones = dynamic_milestones
            self.dynamic_intervals = dynamic_intervals

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect_left(self.dynamic_milestones, progress + 1)
            self.interval = self.dynamic_intervals[step]

    def before_train_epoch(self, runner):
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        model = runner.model
        # JAX-compatible way to handle BatchNorm buffers
        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                if module.track_running_stats:
                    module.running_var = jnp.broadcast_to(module.running_var, (runner.rank,))
                    module.running_mean = jnp.broadcast_to(module.running_mean, (runner.rank,))

        if runner.rank == 0:
            from mmdet.apis import multi_gpu_test
            results = multi_gpu_test(model, self.dataloader, tmpdir=runner.work_dir + '.eval')
            self.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(model, results)
            if self.save_best and key_score is not None:
                self._save_ckpt(runner, key_score)

# Example usage:
# hook = EvalHook(dynamic_intervals=[(1, 5), (10, 15)])