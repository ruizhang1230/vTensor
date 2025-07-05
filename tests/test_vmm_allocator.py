import torch
from vTensor import get_pluggable_allocator


def test_vmm_allocator_basic():
    vmm_allocator = get_pluggable_allocator()
    torch.cuda.change_current_allocator(vmm_allocator)

    #  now all the tensors are allocated via VMM api
    shape = (8 * 1024, 1024)
    x = torch.empty(shape, device="cuda")
    x.zero_()

    y = torch.empty(shape, device="cuda")
    y.zero_()

    y += 1

    output = x + y
    print("deleting x ...")
    del x

    free_bytes = torch.cuda.mem_get_info()[0]

    print(f"free_bytes : {free_bytes}")


def test_vmm_allocator_mapping():
    pass


def test_vmm_allocator_unmapping():
    pass


def test_vmm_allocator_resume():
    pass


def test_vmm_allocator_pause():
    pass


if __name__ == "__main__":
    test_allocator()
    pass
