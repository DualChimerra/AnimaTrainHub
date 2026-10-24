"""H2D-only streaming of frozen, possibly heterogeneous encoder layers.

Unlike the training DiT ring this runs only under no_grad. CPU masters stay
unchanged and a layer's GPU copies are released immediately after its forward.
"""
import torch


class FrozenLayerOffload:
    def __init__(self, layers, count, device):
        self.device = torch.device(device)
        self.handles = []
        self.masters = []
        if count < 0 or count > len(layers):
            raise ValueError(f'Encoder layer offloading must be between 0 and {len(layers)}')
        for layer in list(layers)[len(layers) - count:] if count else []:
            if any(p.requires_grad for p in layer.parameters()):
                raise ValueError('Text encoder layer offloading requires frozen weights')
            masters = [(p, p.detach().to('cpu')) for p in layer.parameters()]
            self.masters.extend(masters)
            for parameter, master in masters:
                parameter.data = master

            def before(module, args, owned=masters):
                if torch.is_grad_enabled():
                    raise RuntimeError('Frozen encoder offloading must run under torch.no_grad()')
                for parameter, master in owned:
                    parameter.data = master.to(self.device)

            def after(module, args, output, owned=masters):
                for parameter, master in owned:
                    parameter.data = master

            self.handles.append(layer.register_forward_pre_hook(before))
            self.handles.append(layer.register_forward_hook(after, always_call=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for parameter, master in self.masters:
            parameter.data = master
        self.masters.clear()
