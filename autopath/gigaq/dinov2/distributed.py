import torch.distributed
import dinov2.distributed

#TODO: spell out args and kwargs and emable multigpu/multinode training
#TODO: pass distributed initialization params through ssl.yml --> cfg
def initialize(*args, **kwargs):
    dinov2.distributed.enable(*args, **kwargs)

def tear_down():
    torch.distributed.destroy_process_group()
