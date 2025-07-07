import torch
import vTensor
from vTensor import get_pluggable_allocator


def test_vmm_allocator_basic():
    vmm_allocator_torch_api = get_pluggable_allocator()
    torch.cuda.change_current_allocator(vmm_allocator_torch_api)

    #  now all the tensors are allocated via VMM api by default
    shape = (8 * 1024, 1024)
    x = torch.empty(shape, dtype=torch.float16, device="cuda")
    x.zero_()

    y = torch.empty(shape, dtype=torch.float16, device="cuda")
    y.zero_()

    y += 1

    output = x + y
    print("deleting x ...")
    del x

    free_bytes = torch.cuda.mem_get_info()[0]

    print(f"free_bytes : {free_bytes}")


def test_vmm_allocator_manually_remapping():
    vmm_allocator_torch_api = get_pluggable_allocator()
    torch.cuda.change_current_allocator(vmm_allocator_torch_api)

    #  now all the tensors are allocated via VMM api by default
    shape = (8 * 1024, 1024)
    x = torch.empty(shape, dtype=torch.float, device="cuda")
    x.zero_()

    # we want to resuse the memroy allocated for x
    new_tensor_dtype = torch.float16

    stream = 0  # torch.cuda.current_stream()
    y = vTensor.vmm_tensor(
        x.data_ptr(),
        x.shape,
        x.stride(),
        new_tensor_dtype,
        x.numel(),
        x.get_device(),
        stream,
    )
    y.zero_()

    y += 1

    z = vTensor.vmm_tensor(
        x.data_ptr(),
        new_tensor_shape,
        new_tensor_stride,
        new_tensor_dtype,
        numel,
        x.get_device(),
        stream,
    )
    z.zero_()

    output = z + y

    print("deleting x ...")
    del x

    y += 1
    pass


def test_vmm_allocator_auto_remapping():
    vmm_allocator_torch_api = get_pluggable_allocator()
    torch.cuda.change_current_allocator(vmm_allocator_torch_api)

    #  now all the tensors are allocated via VMM api by default
    shape = (8 * 1024, 1024)
    x = torch.empty(shape, dtype=torch.float16, device="cuda")
    x.zero_()

    x += 1

    y = torch.empty(shape, dtype=torch.float16, device="cuda")
    y.zero_()

    y += 1

    stream = 0  # torch.cuda.current_stream()

    gpu_v_addr = x.data_ptr()

    print("gpu_v_addr : ", gpu_v_addr)

    vTensor.vmm_dealloc(gpu_v_addr, x.numel(), x.get_device(), stream)
    del x

    z = torch.empty(shape, dtype=torch.float16, device="cuda")

    assert torch.allclose(
        y, z, atol=0, rtol=0
    ), f"z and y should have the same values, even though z is not initialized in pytorch side."

    print(f"✅ Reuse the memory of pre-allocated tensor successuflly")


def test_vmm_allocator_resume():
    pass


def test_vmm_allocator_pause():
    pass


if __name__ == "__main__":
    test_vmm_allocator_auto_remapping()
    pass
