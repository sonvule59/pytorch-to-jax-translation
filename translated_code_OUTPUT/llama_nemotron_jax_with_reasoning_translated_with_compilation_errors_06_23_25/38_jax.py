import numpy as np
import torch

import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

def process_image(img):
    img = cv2.imread(img).astype("float32")
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    img = torch.unsqueeze(img, dim=0)

    return img

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

def generate_jit_model(model_type):
    model_class = torchvision.models.detection.fasterrcnn_resnet50_fpn if model_type == 1 else torchvision.models.detection.maskrcnn_resnet50_fpn
    model = TraceWrapper(model_class(pretrained=True, rpn_pre_nms_top_n_test=1000))
    model.eval()
    inp = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        out = model(inp)
    return model

def test():
    img_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/road_small.jpg"
    download(img_url)

    input_shape = (1, 3, 300, 300)
    scripted_model = generate_jit_model(1)
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list=[input_shape])

    data = process_image(img)
    data_np = data.detach().numpy()

    with torch.no_grad():
        pt_res = scripted_model(data)

    for target in ["cuda", "cpu"]:
        try:
            dev = tvm.device(target)
        except:
            continue
        with tvm.transform.PassContext(opt_level=3):
            vm_exec = relay.vm.compile(mod, target=target, params=params)
        vm = VirtualMachine(vm_exec, dev)
        vm.set_input("main", **{f"input{0}": data_np})
        tvm_res = vm.run()

        if target == "cpu":
            assert tvm.testing.assert_allclose(pt_res.cpu().numpy(), tvm_res[0].numpy(), rtol=1e-5, atol=1e-5)
            assert tvm.testing.assert_allclose(pt_res[1].cpu().numpy(), tvm_res[1].numpy(), rtol=1e-5, atol=1e-5)
            assert np.testing.assert_equal(pt_res[2].cpu().numpy(), tvm_res[2].numpy())
        else:
            assert len(pt_res[0]) == len(tvm_res[0])
            assert len(pt_res[1]) == len(tvm_res[1])
            assert np.testing.assert_allclose(pt_res[1].cpu(), tvm_res[1].cpu(), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test()