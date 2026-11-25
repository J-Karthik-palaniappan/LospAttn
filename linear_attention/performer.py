import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from einops import repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim = -2)

class FastAttention(nn.Module):
    def __init__(self, nb_features, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), no_projection=False,):
        super().__init__()

        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

        self.register_buffer("projection_matrix", None)

        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = causal_linear_attention
            except ImportError:
                print("CUDA causal kernel unavailable; using PyTorch fallback.")
                self.causal_linear_fn = causal_linear_attention_noncuda

    def _init_projection(self, dim_heads, device):
        self.projection_matrix = gaussian_orthogonal_random_matrix(
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=self.ortho_scaling,
            device=device,
        )

    @torch.no_grad()
    def redraw_projection_matrix(self, device=None):
        if self.projection_matrix is None:
            return

        device = device or self.projection_matrix.device
        projection = gaussian_orthogonal_random_matrix(nb_rows=self.nb_features, nb_columns=self.projection_matrix.shape[1], scaling=self.ortho_scaling, device=device,)
        self.projection_matrix.copy_(projection)

    def forward(self, q, k, v):
        dim_heads = q.shape[-1]

        if not self.no_projection:
            if self.projection_matrix is None:
                self._init_projection(dim_heads, q.device)
            elif self.projection_matrix.shape[1] != dim_heads:
                raise ValueError("Input head dimension differs from initialized projection.")

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn, projection_matrix=self.projection_matrix,)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix,)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        return attn_fn(q, k, v)