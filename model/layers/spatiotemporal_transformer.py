"""Codes adapted from TSL (Torch SpatioTemporal)."""

from functools import partial
from typing import Optional
import torch
import torch.nn.functional as F
import torch_geometric
from einops import rearrange
from torch import Tensor, nn


@torch.jit.script
def _get_causal_mask(seq_len: int, diagonal: int = 0,
                     device: Optional[torch.device] = None):
    # mask keeping only previous steps
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    causal_mask = torch.triu(ones, diagonal)
    return causal_mask


_torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity'
}


def _identity(x):
    return x


def get_layer_activation(activation: Optional[str] = None):
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation in _torch_activations_dict:
        return getattr(nn, _torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")


def get_functional_activation(activation: Optional[str] = None):
    if activation is None:
        return _identity
    activation = activation.lower()
    if activation == 'linear':
        return _identity
    if activation in ['tanh', 'sigmoid']:
        return getattr(torch, activation)
    if activation in _torch_activations_dict:
        return getattr(F, activation)
    raise ValueError(f"Activation '{activation}' not valid.")


class LayerNorm(torch.nn.Module):
    r"""Applies layer normalization.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.ones(self.weight)
        torch_geometric.nn.inits.zeros(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, unbiased=False, keepdim=True)

        out = (x - mean) / (std + self.eps)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class MultiHeadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, heads,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 axis='time',
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 device=None,
                 dtype=None,
                 causal=False) -> None:
        if axis in ['time', 0]:
            shape = 's (b n) c'
        elif axis in ['nodes', 1]:
            if causal:
                raise ValueError(f'Cannot use causal attention for axis "{axis}".')
            shape = 'n (b s) c'
        else:
            raise ValueError("Axis can either be 'time' (0) or 'nodes' (1), "
                             f"not '{axis}'.")
        self._in_pattern = f'b s n c -> {shape}'
        self._out_pattern = f'{shape} -> b s n c'
        self.causal = causal
        # Impose batch dimension as the second one
        super(MultiHeadAttention, self).__init__(embed_dim, heads,
                                                 dropout=dropout,
                                                 bias=bias,
                                                 add_bias_kv=add_bias_kv,
                                                 add_zero_attn=add_zero_attn,
                                                 kdim=kdim,
                                                 vdim=vdim,
                                                 batch_first=False,
                                                 device=device,
                                                 dtype=dtype)
        # change projections
        if qdim is not None and qdim != embed_dim:
            self.qdim = qdim
            self.q_proj = torch_geometric.nn.dense.Linear(self.qdim, embed_dim)
        else:
            self.qdim = embed_dim
            self.q_proj = nn.Identity()

    def forward(self, query: Tensor,
                key: torch_geometric.typing.OptTensor = None,
                value: torch_geometric.typing.OptTensor = None,
                key_padding_mask: torch_geometric.typing.OptTensor = None,
                need_weights: bool = True,
                attn_mask: torch_geometric.typing.OptTensor = None):
        # inputs: [batches, steps, nodes, channels] -> [s (b n) c]
        if key is None:
            key = query
        if value is None:
            value = key
        batch = value.shape[0]
        query, key, value = [rearrange(x, self._in_pattern)
                             for x in (query, key, value)]

        if self.causal:
            causal_mask = _get_causal_mask(key.size(0), diagonal=1, device=query.device)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                attn_mask = torch.logical_and(attn_mask, causal_mask)
        attn_output, attn_weights = super(MultiHeadAttention,
                                          self).forward(self.q_proj(query),
                                                        key,
                                                        value,
                                                        key_padding_mask,
                                                        need_weights,
                                                        attn_mask)
        attn_output = rearrange(attn_output, self._out_pattern, b=batch) \
            .contiguous()
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights, '(b d) l m -> b d l m',
                                     b=batch).contiguous()
        return attn_output, attn_weights


class TransformerLayer(nn.Module):
    r"""A Transformer layer from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017).

    This layer can be instantiated to attend the temporal or spatial dimension.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        axis (str, optional): Dimension on which to apply attention to update
            the representations. Can be either, 'time' or 'nodes'.
            (default: :obj:`'time'`)
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention (has an effect only if :attr:`axis` is
            :obj:`'time'`). (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 axis='time',
                 causal=True,
                 activation='elu',
                 dropout=0.,
                 return_attention=False):
        super(TransformerLayer, self).__init__()
        self.att = MultiHeadAttention(embed_dim=hidden_size,
                                      qdim=input_size,
                                      kdim=input_size,
                                      vdim=input_size,
                                      heads=n_heads,
                                      axis=axis,
                                      causal=causal)

        self.return_att = return_attention

        if input_size != hidden_size:
            self.skip_conn = nn.Linear(input_size, hidden_size)
        else:
            self.skip_conn = nn.Identity()

        self.norm1 = LayerNorm(input_size)

        self.mlp = nn.Sequential(LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ff_size),
                                 get_layer_activation(activation)(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ff_size, hidden_size),
                                 nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

        self.activation = get_functional_activation(activation)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """"""
        # x: [batch, steps, nodes, features]
        '''x = self.skip_conn(x) + self.dropout(
            self.att(self.norm1(x), attn_mask=mask)[0])'''
        x_skip = self.skip_conn(x)
        x_att, att_weights = self.att(self.norm1(x), attn_mask=mask)
        x = x_skip + self.dropout(x_att)
        x = x + self.mlp(x)
        if self.return_att:
            return x, att_weights
        return x


class SpatioTemporalTransformerLayer(nn.Module):
    r"""A :class:`~tsl.nn.blocks.encoders.TransformerLayer` which attend both
    the spatial and temporal dimensions by stacking two
    :class:`~tsl.nn.layers.base.MultiHeadAttention` layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention.
            (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.,
                 return_attention=False):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.temporal_att = MultiHeadAttention(embed_dim=hidden_size,
                                               qdim=input_size,
                                               kdim=input_size,
                                               vdim=input_size,
                                               heads=n_heads,
                                               axis='time',
                                               causal=causal)

        self.spatial_att = MultiHeadAttention(embed_dim=hidden_size,
                                              qdim=hidden_size,
                                              kdim=hidden_size,
                                              vdim=hidden_size,
                                              heads=n_heads,
                                              axis='nodes',
                                              causal=False)

        self.skip_conn = nn.Linear(input_size, hidden_size)

        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(hidden_size)

        self.mlp = nn.Sequential(LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ff_size),
                                 get_layer_activation(activation)(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ff_size, hidden_size),
                                 nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

        self.return_attention = return_attention

    '''def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """"""
        # x: [batch, steps, nodes, features]
        x = self.skip_conn(x) + self.dropout(
            self.temporal_att(self.norm1(x), attn_mask=mask)[0])
        x = x + self.dropout(
            self.spatial_att(self.norm2(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x'''

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """"""
        # x: [batch, steps, nodes, features]
        tatt_out, temporal_att = self.temporal_att(self.norm1(x), attn_mask=mask)
        x = self.skip_conn(x) + self.dropout(
            tatt_out)
        satt_out, spatial_att = self.spatial_att(self.norm2(x), attn_mask=mask)
        x = x + self.dropout(
            satt_out)
        x = x + self.mlp(x)
        if self.return_attention:
            return x, (temporal_att, spatial_att)
        else:
            return x


class SpatioTemporalTransformer(nn.Module):
    r"""A stack of Transformer layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        output_size (int, optional): Size of an optional linear readout.
        n_layers (int, optional): Number of Transformer layers.
        n_heads (int, optional): Number of parallel attention heads.
        axis (str, optional): Dimension on which to apply attention to update
            the representations. Can be either, 'time', 'nodes', or 'both'.
            (default: :obj:`'time'`)
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention (has an effect only if :attr:`axis` is
            :obj:`'time'` or :obj:`'both'`).
            (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 output_size=None,
                 n_layers=1,
                 n_heads=1,
                 axis='time',
                 causal=True,
                 activation='elu',
                 dropout=0.,
                 return_attention=False):
        super(SpatioTemporalTransformer, self).__init__()
        self.f = getattr(F, activation)
        if ff_size is None:
            ff_size = hidden_size

        if axis in ['time', 'nodes']:
            transformer_layer = partial(TransformerLayer, axis=axis)
        elif axis == 'both':
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        layers = []
        for i in range(n_layers):
            layers.append(
                transformer_layer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    ff_size=ff_size,
                    n_heads=n_heads,
                    causal=causal,
                    activation=activation,
                    dropout=dropout,
                    return_attention=return_attention))

        self.net = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor):
        """"""
        x = self.net(x)
        if self.readout is not None:
            return self.readout(x)
        return x