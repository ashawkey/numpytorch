from numpy.core.numeric import allclose
import torch
import numpytorch as np
#np.list_patches()

def test_cat():
    a = np.randn(2, 3, 1)
    assert np.allclose(np.cat([a, a]), torch.cat([torch.from_numpy(a), torch.from_numpy(a)]).numpy())
    assert np.allclose(np.cat([a, a], dim=-1), torch.cat([torch.from_numpy(a), torch.from_numpy(a)], dim=-1).numpy())


def test_chunk():
    a = np.randn(10,1)
    x, y, z = np.chunk(a, 3, dim=-2)
    nx, ny, nz = torch.chunk(torch.from_numpy(a), 3, dim=-2)
    assert np.allclose(x, nx.numpy())
    assert np.allclose(y, ny.numpy())
    assert np.allclose(z, nz.numpy())

def test_gather():
    a = np.arange(9).reshape(3, 3)
    b = np.array([[2, 1], [0, 1]])
    assert np.allclose(
        np.gather(a, 1, b),
        torch.gather(torch.from_numpy(a), 1, torch.from_numpy(b)).numpy()
    )

def test_scatter_add():
    input = torch.zeros(2, 2).long()
    src = torch.ones(2, 3).long()
    # we want input[:, 0] = src[:, 0]; input[:, 1] = src[:, 1] + src[:, 2]
    index = torch.LongTensor([0, 1, 1]) 
    # dim=1 since we collect from column(dim1).
    # we have to EXPLICITLY repeat index, since torch only use src[:index.shape] !!!
    assert np.allclose(
        torch.scatter_add(input, 1, index[None, :].repeat(2, 1), src).numpy(), # [[1,2], [1,2]]
        np.scatter_add(input.numpy(), 1, index[None, :].repeat(2, 1).numpy(), src.numpy())
    )
    assert np.allclose(
        torch.scatter_add(input, 1, index[None, :], src).numpy(), # [[1,2], [1,2]]
        np.scatter_add(input.numpy(), 1, index[None, :].numpy(), src.numpy())
    )
    
    
def test_narrow():
    a = np.randn(3,4,5)
    assert np.allclose(
        np.narrow(a, 2, 1, 4), 
        torch.narrow(torch.from_numpy(a), 2, 1, 4).numpy()
    )

def test_nonzero():
    a = np.randn(3,3)
    a[a > 0] = 0
    assert np.allclose(np.torch_nonzero(a, as_tuple=False), torch.nonzero(torch.from_numpy(a), as_tuple=False).numpy())

def test_view():
    a = np.arange(6).torch_view(2, 3)
    na = torch.arange(6).view(2, 3)
    assert np.allclose(a, na.numpy())
    assert np.allclose(a.shape, na.numpy().shape)
    a = a.torch_view(-1)
    na = na.view(-1)
    assert np.allclose(a, na.numpy())
    assert np.allclose(a.shape, na.numpy().shape)


def test_squeeze():
    a = np.arange(6).unsqueeze(0)
    na = torch.arange(6).unsqueeze(0)
    assert np.allclose(a, na.numpy())
    a = a.unsqueeze(-1)
    na = na.unsqueeze(-1)
    assert np.allclose(a, na.numpy())
    a = a.squeeze(-1)
    na = na.squeeze(-1)
    assert np.allclose(a, na.numpy())


def test_chain():
    a = np.arange(3).float().add(-1)
    na = torch.arange(3).float().add(-1)
    assert np.allclose(a, na.numpy())
    a = a.abs()
    na = na.abs()
    assert np.allclose(a, na.numpy())
    a = a.sin()
    na = na.sin()
    assert np.allclose(a, na.numpy())

def test_meshgrid():
    nx, ny = torch.meshgrid(torch.arange(3), torch.arange(4))
    x, y = np.torch_meshgrid(np.arange(3), np.arange(4))
    assert np.allclose(nx.numpy(), x)
    assert np.allclose(ny.numpy(), y)

def test_repeat():
    a = np.arange(4).torch_view(1,1,2,2).torch_repeat(3,1,2,2)
    na = torch.arange(4).view(1,1,2,2).repeat(3,1,2,2)
    assert np.allclose(a, na.numpy())

def test_permute():
    a = np.arange(4).torch_view(1,1,2,2).permute(3,2,0,1)
    na = torch.arange(4).view(1,1,2,2).permute(3,2,0,1)
    assert np.allclose(a, na.numpy())

def test_type():
    a = np.arange(3)
    na = torch.arange(3)
    assert np.allclose(a.float(), na.float().numpy())
    assert np.allclose(a.long(), na.long().numpy())
    assert np.allclose(a.int(), na.int().numpy())
