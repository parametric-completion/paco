import torch
import torch.nn as nn
from timm.models.layers import DropPath
from pointnet2_ops import pointnet2_utils
import einops


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Find K nearest neighbors.
    
    Args:
        nsample: Maximum sample number in local region
        xyz: All points, shape [B, N, C]
        new_xyz: Query points, shape [B, S, C]
        
    Returns:
        group_idx: Grouped points indices, shape [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared Euclidean distance between each pair of points.
    
    For point sets with coordinates (x, y, z), the formula is:
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src^2, dim=-1) + sum(dst^2, dim=-1) - 2 * src^T * dst
    
    Args:
        src: Source points, shape [B, N, C]
        dst: Target points, shape [B, M, C]
        
    Returns:
        dist: Per-point squared distance, shape [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Sample points by index.
    
    Args:
        points: Input points data, shape [B, N, C]
        idx: Sample index data, shape [B, S] or [B, S, K]
        
    Returns:
        new_points: Indexed points data, shape [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class MLP(nn.Module):
    """Multi-layer perceptron module.
    
    A simple MLP with configurable hidden dimension and dropout.
    """
    
    def __init__(self, in_features: int, hidden_features: int = None, 
                out_features: int = None, act_layer=nn.GELU, drop: float = 0.):
        """Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension (defaults to in_features)
            out_features: Output feature dimension (defaults to in_features)
            act_layer: Activation layer type
            drop: Dropout probability
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module.
    
    Standard implementation of multi-head self-attention with optional masking.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0.):
        """Initialize attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in query, key, value projections
            qk_scale: Scale factor for attention (defaults to 1/sqrt(head_dim))
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for projection
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute attention.
        
        Args:
            x: Input tensor, shape [B, N, C]
            mask: Optional mask tensor, shape [B, N, N]
            
        Returns:
            Output tensor after attention, shape [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            mask = (mask > 0)  # convert to boolen, shape torch.BoolTensor[N, N]
            attn = attn.masked_fill(mask, mask_value)  # B h N N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention module.
    
    Performs attention between query and key-value pairs from different sources.
    """
    
    def __init__(self, dim: int, out_dim: int, num_heads: int = 8, 
                qkv_bias: bool = False, qk_scale: float = None, 
                attn_drop: float = 0., proj_drop: float = 0.):
        """Initialize cross-attention module.
        
        Args:
            dim: Input dimension 
            out_dim: Output dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in query, key, value projections
            qk_scale: Scale factor for attention (defaults to 1/sqrt(head_dim))
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for projection
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention.
        
        Args:
            q: Query tensor, shape [B, N, C]
            v: Key-value tensor, shape [B, M, C]
            
        Returns:
            Output tensor after cross-attention, shape [B, N, C]
        """
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DeformableLocalAttention(nn.Module):
    """DeformableLocalAttention for self-attention.
    
    Queries a local region for each token and performs self-attention
    among them, using the region features after maxpooling to update
    the token features.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0.,
                k: int = 10, n_group: int = 2):
        """Initialize deformable local attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in query, key, value projections
            qk_scale: Scale factor for attention (defaults to 1/sqrt(head_dim))
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for projection
            k: Number of neighbors to consider
            n_group: Number of groups for channel-wise splitting
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            pos: Position tensor, shape [B, N, 3]
            idx: Optional index tensor, shape [B, N, k]
            
        Returns:
            Output tensor after deformable local attention, shape [B, N, C]
        """
        B, N, C = x.shape

        if idx is None:
            idx = knn_point(self.k, pos, pos)  # B N k

        q = self.proj_q(x)
        v_off = self.proj_v_off(x)
        local_v = index_points(v_off, idx)  # B N k C
        off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group,
                                       c=self.group_dims)  # Bg N k c
        group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c
        shift_feat = torch.cat([
            off_local_v,
            group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
        ], dim=-1)  # Bg N k 2c
        offset = self.linear_offset(shift_feat)  # Bg N k 3
        offset = offset.tanh()  # Bg N k 3
        local_v_pos = index_points(pos, idx)  # B N k 3
        local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)  # B g N k 3
        local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c')  # Bg N k 3
        shift_pos = local_v_pos + offset  # Bg N 2*k 3
        shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c')  # Bg k*N 3
        pos = pos.unsqueeze(1).expand(-1, self.n_group, -1, -1)  # B g N 3
        pos = einops.rearrange(pos, 'b g n c -> (b g) n c')  # Bg N 3
        v = einops.rearrange(x, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c
        dist, _idx = pointnet2_utils.three_nn(shift_pos.contiguous(), pos.contiguous())  # Bg k*N 3, Bg k*N 3
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), _idx,
                                                               weight).transpose(-1, -2).contiguous()
        interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group,
                                              n=N, k=self.k)  # B N k gc

        local_q = index_points(q, idx)  # B N k C
        q = einops.rearrange(local_q, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
        k = self.proj_k(interpolated_feats)
        k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
        v = self.proj_v(interpolated_feats)
        v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c

        attn = torch.einsum('b m c, b n c -> b m n', q, k)  # BHN, k, k
        attn = attn.mul(self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b n c -> b m c', attn, v)  # BHN k c
        out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads)  # B N k C
        out = out.max(dim=2, keepdim=False)[0]  # B N C
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class DeformableLocalCrossAttention(nn.Module):
    """DeformableLocalAttention for self-attn or cross-attn.
    
    Queries a local region for each token and performs cross-attention
    among query token and local region.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0., 
                k: int = 10, n_group: int = 2):
        """Initialize deformable local cross-attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in query, key, value projections
            qk_scale: Scale factor for attention (defaults to 1/sqrt(head_dim))
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for projection
            k: Number of neighbors to consider
            n_group: Number of groups for channel-wise splitting
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.k = k
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, q: torch.Tensor, q_pos: torch.Tensor, v: torch.Tensor = None, 
                v_pos: torch.Tensor = None, idx: torch.Tensor = None, denoise_length: int = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v: Key-value tensor, shape [B, M, C]
            v_pos: Key-value position tensor, shape [B, M, 3]
            idx: Optional index tensor, shape [B, N, k]
            denoise_length: Optional denoise length
            
        Returns:
            Output tensor after deformable local cross-attention, shape [B, N, C]
        """
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            k = v
            NK = k.size(1)

            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # B N k

            q = self.proj_q(q)
            v_off = self.proj_v_off(v)
            local_v = index_points(v_off, idx)  # B N k C
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group,
                                           c=self.group_dims)  # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset = self.linear_offset(shift_feat)  # Bg N k 3
            offset = offset.tanh()  # Bg N k 3
            local_v_pos = index_points(v_pos, idx)  # B N k 3
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)  # B g N k 3
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c')  # Bg N k 3
            shift_pos = local_v_pos + offset  # Bg N k 3
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c')  # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1)  # B g Nk 3
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c')  # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg Nk c
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B,
                                                  g=self.n_group, n=N, k=self.k)  # B N k gc

            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(
                -2)  # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k)  # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v)  # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads)  # B N 1 C
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

        else:
            v = q
            v_pos = q_pos

            B, N, C = q.shape

            q = self.proj_q(q)
            v_off = self.proj_v_off(v)

            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k

            local_v_r = index_points(v_off[:, :-denoise_length], idx)  # B N_r k C
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx)  # B N_r k 3

            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k

            local_v_n = index_points(v_off, idx)  # B N_n k C
            local_v_n_pos = index_points(v_pos, idx)  # B N_n k 3

            local_v = torch.cat([local_v_r, local_v_n], dim=1)  # B N k C

            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group,
                                           c=self.group_dims)  # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c

            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset = self.linear_offset(shift_feat)  # Bg N k 3
            offset = offset.tanh()  # Bg N k 3
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)  # B g N k 3
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c')  # Bg N k 3
            shift_pos = local_v_pos + offset  # Bg N k 3
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c')  # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1)  # B g Nk 3
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c')  # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg Nk c
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B,
                                                  g=self.n_group, n=N, k=self.k)  # B N k gc

            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(
                -2)  # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k)  # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v)  # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads)  # B N 1 C
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)
        return out


class ImprovedDeformableLocalCrossAttention(nn.Module):
    """DeformableLocalAttention for self-attn or cross-attn.
    
    Queries a local region for each token and performs cross-attention
    among query token and local region.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0., 
                k: int = 10, n_group: int = 2):
        """Initialize improved deformable local cross-attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in query, key, value projections
            qk_scale: Scale factor for attention (defaults to 1/sqrt(head_dim))
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for projection
            k: Number of neighbors to consider
            n_group: Number of groups for channel-wise splitting
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, q: torch.Tensor, q_pos: torch.Tensor, v: torch.Tensor = None, v_pos: torch.Tensor = None, idx: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v: Key-value tensor, shape [B, M, C]
            v_pos: Key-value position tensor, shape [B, M, 3]
            idx: Optional index tensor, shape [B, N, k]
            
        Returns:
            Output tensor after improved deformable local cross-attention, shape [B, N, C]
        """
        if v is None:
            v = q
        if v_pos is None:
            v_pos = q_pos

        B, N, C = q.shape
        k = v
        NK = k.size(1)

        if idx is None:
            idx = knn_point(self.k, v_pos, q_pos)  # B N k

        q = self.proj_q(q)
        v_off = self.proj_v_off(v)
        local_v = index_points(v_off, idx)  # B N k C
        off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group,
                                       c=self.group_dims)  # Bg N k c
        group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c
        shift_feat = torch.cat([
            off_local_v,
            group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
        ], dim=-1)  # Bg N k 2c
        offset = self.linear_offset(shift_feat)  # Bg N k 3
        offset = offset.tanh()  # Bg N k 3

        local_v_pos = index_points(v_pos, idx)  # B N k 3
        local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)  # B g N k 3
        local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c')  # Bg N k 3

        scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0]  # Bg N 3
        scale = scale.unsqueeze(-2) * 0.5  # Bg N 1 3
        shift_pos = local_v_pos + offset * scale  # Bg N k 3

        shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c')  # Bg k*N 3
        v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1)  # B g Nk 3
        v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c')  # Bg Nk 3
        v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg Nk c
        dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # Bg k*N 3, Bg k*N 3
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(
            -1, -2).contiguous()
        interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group,
                                              n=N, k=self.k)  # B N k gc

        q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(-2)  # BHN 1 c
        k = self.proj_k(interpolated_feats)
        k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
        v = self.proj_v(interpolated_feats)
        v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c

        attn = torch.einsum('b m c, b n c -> b m n', q, k)  # BHN, 1, k
        attn = attn.mul(self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b n c -> b m c', attn, v)  # BHN 1 c
        out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads)  # B N 1 C
        out = out.squeeze(2)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class ImprovedDeformableLocalGraphAttention(nn.Module):
    """DeformableLocalAttention for self-attn or cross-attn.
    
    Queries a local region for each token and performs graph convolution
    among query token and local region.
    """
    
    def __init__(self, dim: int, k: int = 10):
        """Initialize improved deformable local graph attention module.
        
        Args:
            dim: Input dimension
            k: Number of neighbors to consider
        """
        super().__init__()
        self.dim = dim

        self.proj_v_off = nn.Linear(dim, dim)

        self.k = k  # To be controlled 
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q: torch.Tensor, q_pos: torch.Tensor, v: torch.Tensor = None, 
                v_pos: torch.Tensor = None, idx: torch.Tensor = None, denoise_length: int = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v: Key-value tensor, shape [B, M, C]
            v_pos: Key-value position tensor, shape [B, M, 3]
            idx: Optional index tensor, shape [B, N, k]
            denoise_length: Optional denoise length
            
        Returns:
            Output tensor after improved deformable local graph attention, shape [B, N, C]
        """
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape

            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # B N k
            v_off = self.proj_v_off(v)
            off_local_v = index_points(v_off, idx)  # B N k C
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset = self.linear_offset(shift_feat)  # B N k 3
            offset = offset.tanh()  # B N k 3

            local_v_pos = index_points(v_pos, idx)  # B N k 3

            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0]  # B N 3
            scale = scale.unsqueeze(-2) * 0.5  # B N 1 3
            shift_pos = local_v_pos + offset * scale  # B N k 3

            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c')  # B k*N 3
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k)  # B N k c

            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C
        else:
            v = q
            v_pos = q_pos

            B, N, C = q.shape

            v_off = self.proj_v_off(v)

            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k

            local_v_r_off = index_points(v_off[:, :-denoise_length], idx)  # B N_r k C
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx)  # B N_r k 3

            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k

            local_v_n_off = index_points(v_off, idx)  # B N_n k C
            local_v_n_pos = index_points(v_pos, idx)  # B N_n k 3

            off_local_v = torch.cat([local_v_r_off, local_v_n_off], dim=1)  # B N k C
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset = self.linear_offset(shift_feat)  # B N k 3
            offset = offset.tanh()  # B N k 3

            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3

            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0]  # B N 3
            scale = scale.unsqueeze(-2) * 0.5  # B N 1 3
            shift_pos = local_v_pos + offset * scale  # B N k 3

            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c')  # B k*N 3
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k)  # B N k c

            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C

        return out


class DynamicGraphAttention(nn.Module):
    """DynamicGraphAttention for self-attn or cross-attn.
    
    Queries a local region for each token and performs Conv2d with maxpooling
    to build the graph feature for each token.
    """
    
    def __init__(self, dim: int, k: int = 10):
        """Initialize dynamic graph attention module.
        
        Args:
            dim: Input dimension
            k: Number of neighbors to consider
        """
        super().__init__()
        self.dim = dim
        self.k = k
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q: torch.Tensor, q_pos: torch.Tensor, v: torch.Tensor = None, 
                v_pos: torch.Tensor = None, idx: torch.Tensor = None, denoise_length: int = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v: Key-value tensor, shape [B, M, C]
            v_pos: Key-value position tensor, shape [B, M, 3]
            idx: Optional index tensor, shape [B, N, k]
            denoise_length: Optional denoise length
            
        Returns:
            Output tensor after dynamic graph attention, shape [B, N, C]
        """
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
            B, N, C = q.shape
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # B N k
            local_v = index_points(v, idx)  # B N k C
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((local_v - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C
        else:
            v = q
            v_pos = q_pos
            B, N, C = q.shape
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k
            local_v_r = index_points(v[:, :-denoise_length], idx)  # B N_r k C

            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k

            local_v_n = index_points(v, idx)  # B N_n k C

            local_v = torch.cat([local_v_r, local_v_n], dim=1)
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((local_v - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C

        return out


class LayerScale(nn.Module):
    """Layer scaling module."""
    
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        """Initialize layer scaling module.
        
        Args:
            dim: Input dimension
            init_values: Initial scaling values
            inplace: Whether to perform in-place scaling
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            
        Returns:
            Scaled tensor, shape [B, N, C]
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    """Normal self-attention block."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                drop: float = 0., attn_drop: float = 0., init_values: float = None, 
                drop_path: float = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            pos: Position tensor, shape [B, N, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class RegionWiseBlock(nn.Module):
    """Block with region-wise deformable attention.
    
    Uses maxpooling for token feature update.
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                drop: float = 0., attn_drop: float = 0., init_values: float = None, 
                drop_path: float = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize region-wise block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.deformable_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                        attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            pos: Position tensor, shape [B, N, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        x = x + self.drop_path1(self.ls1(self.deformable_attn(self.norm1(x), pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm1(x))))
        return x


class DeformableAttnBlock(nn.Module):
    """Deformable attention block."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                drop: float = 0., attn_drop: float = 0., init_values: float = None, 
                drop_path: float = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize deformable attention block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.deformable_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                             attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            pos: Position tensor, shape [B, N, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        x = x + self.drop_path1(self.ls1(self.deformable_attn(self.norm1(x), pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class GraphConvBlock(nn.Module):
    """Graph convolution block."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                drop: float = 0., attn_drop: float = 0., init_values: float = None, 
                drop_path: float = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize graph convolution block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.graphattn = DynamicGraphAttention(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            pos: Position tensor, shape [B, N, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        x = x + self.drop_path1(self.ls1(self.graphattn(self.norm1(x), pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DecoderBlock(nn.Module):
    """Decoder block."""
    
    def __init__(self, dim: int, num_heads: int, dim_q: int = None, mlp_ratio: float = 4., 
                qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0., 
                init_values: float = None, drop_path: float = 0., act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm):
        """Initialize decoder block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dim_q: Query dimension (defaults to input dimension)
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, q: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor, v_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            v: Key-value tensor, shape [B, M, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v_pos: Key-value position tensor, shape [B, M, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.attn(self.norm_q(q), self.norm_v(v))))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm2(q))))
        return q


class DeformableAttnDecoderBlock(nn.Module):
    """Deformable attention decoder block."""
    
    def __init__(self, dim: int, num_heads: int, dim_q: int = None, mlp_ratio: float = 4., 
                qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0., 
                init_values: float = None, drop_path: float = 0., act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm):
        """Initialize deformable attention decoder block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dim_q: Query dimension (defaults to input dimension)
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = DeformableLocalCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, q: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor, v_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            v: Key-value tensor, shape [B, M, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v_pos: Key-value position tensor, shape [B, M, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.attn(q=self.norm_q(q), v=self.norm_v(v), q_pos=q_pos, v_pos=v_pos)))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm2(q))))
        return q


class GraphConvDecoderBlock(nn.Module):
    """Graph convolution decoder block."""
    
    def __init__(self, dim: int, num_heads: int, dim_q: int = None, mlp_ratio: float = 4., 
                qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0., 
                init_values: float = None, drop_path: float = 0., act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm):
        """Initialize graph convolution decoder block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dim_q: Query dimension (defaults to input dimension)
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            qkv_bias: Whether to use bias in query, key, value projections
            drop: Dropout rate
            attn_drop: Dropout rate for attention weights
            init_values: Initial scaling values
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = DynamicGraphAttention(dim)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, q: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor, v_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor, shape [B, N, C]
            v: Key-value tensor, shape [B, M, C]
            q_pos: Query position tensor, shape [B, N, 3]
            v_pos: Key-value position tensor, shape [B, M, 3]
            
        Returns:
            Output tensor, shape [B, N, C]
        """
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.attn(q=self.norm_q(q), v=self.norm_v(v), q_pos=q_pos, v_pos=v_pos)))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm2(q))))
        return q
