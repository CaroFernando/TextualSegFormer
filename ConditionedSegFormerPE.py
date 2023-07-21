import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Iterable, List

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
class VisualEmbeddingsMixer(nn.Module):
    def __init__(self, channels, patch_emb_dim, num_heads):
        super().__init__()
        self.patch_embeddings_fcn = nn.Sequential(
            nn.Linear(patch_emb_dim, channels)
        )
        self.att = nn.MultiheadAttention(channels, num_heads = num_heads, batch_first = True)
        # self.batch_norm = nn.BatchNorm2d(channels)
        self.batch_norm = LayerNorm2d(channels)

    def forward(self, x, patch_embeddings):
        _, _, h, w = x.shape
        patch_embeddings = self.patch_embeddings_fcn(patch_embeddings)
        xr = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(xr, patch_embeddings, patch_embeddings)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h = h, w = w)
        return self.batch_norm(out + x)
    
class TextEmbeddingsMixer(nn.Module):
    def __init__(self, channels, embedding_dim, num_heads):
        super().__init__()
        self.condition_emb = nn.Linear(embedding_dim, channels)
        self.att = nn.MultiheadAttention(channels, num_heads = num_heads, batch_first = True)
        # self.batch_norm = nn.BatchNorm2d(channels)
        self.batch_norm = LayerNorm2d(channels)

    def forward(self, x, condition):
        _, _, h, w = x.shape
        condition = self.condition_emb(condition)
        out = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(out, condition, condition)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h = h, w = w)
        return self.batch_norm(out + x)
 
# xout = MLP(GELU(Conv3×3(MLP(xin)))) + xin,
class MixFFN(nn.Module):
    def __init__(self, channels, expansion, residual = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 1),
            nn.Conv2d(channels, channels * expansion, kernel_size = 3, groups = channels, padding = 1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, kernel_size = 1)
        )

        self.norm = nn.BatchNorm2d(channels)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return self.norm(self.mlp(x) + x)
        else:
            return self.norm(self.mlp(x))
    
class OverlapPatchMerging(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size, overlap_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size = patch_size, stride = overlap_size, padding = patch_size // 2, bias = False),
            LayerNorm2d(out_channels)
        )

class SegFormerConditionedEncoderBlock(nn.Module):
    def __init__(self, channels, condition_dim, patch_embedding_dim, num_heads, expansion):
        super().__init__()
        self.mha1 = VisualEmbeddingsMixer(channels, patch_embedding_dim, num_heads)
        self.mha2 = TextEmbeddingsMixer(channels, condition_dim, num_heads)
        self.ffn = MixFFN(channels, expansion)

    def forward(self, x, pe, condition):
        x = self.mha1(x, pe)
        x = self.mha2(x, condition)
        x = self.ffn(x)
        return x
    
class SegFormerEncoderBlock(nn.Module):
    def __init__(
        self, 
        inchannels: int,
        outchannels: int,
        condition_dim: int,
        patch_embedding_dim,
        patch_size: int,
        overlap_size: int,
        depth: int,
        num_heads: int,
        expansion: int
    ):
        super().__init__()
        self.patch_merge = OverlapPatchMerging(inchannels, outchannels, patch_size, overlap_size)
        self.blocks = nn.ModuleList([
            SegFormerConditionedEncoderBlock(
                outchannels, 
                condition_dim,
                patch_embedding_dim,
                num_heads,
                expansion
            ) for _ in range(depth)
        ])

    def forward(self, x, pe, condition):
        x = self.patch_merge(x)
        for block in self.blocks:
            x = block(x, pe, condition)
        return x

class SegFormerEncoder(nn.Module):
    def __init__(
        self, 
        inchannels: int,
        widths: List[int],
        depths: List[int],
        condition_dim: int,
        patch_embedding_dim: int,
        patch_sizes: List[int],
        overlap_sizes: List[int],
        num_heads: List[int],
        expansions: List[int]
    ):
        super().__init__()
        all_channels = [inchannels] + widths
        input_channels = all_channels[:-1]
        output_channels = all_channels[1:]

        self.blocks = nn.ModuleList([
            SegFormerEncoderBlock(
                inchannels = inchannels,
                outchannels = outchannels,
                condition_dim = condition_dim,
                patch_embedding_dim = patch_embedding_dim,
                patch_size = patch_size,
                overlap_size = overlap_size,
                depth = depth,
                num_heads = num_head,
                expansion = expansion
            ) for inchannels, outchannels, patch_size, overlap_size, depth, num_head, expansion in zip(
                input_channels,
                output_channels,
                patch_sizes,
                overlap_sizes,
                depths,
                num_heads,
                expansions
            )
        ])

    def forward(self, x, pe, condition):
        outs = []
        for block in self.blocks:
            x = block(x, pe, condition)
            outs.append(x)
        return outs
    
class SegFormerDecoderBlock(nn.Module):
    def __init__(self, inchannels, outchannels, scalefactor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = scalefactor, mode = "bilinear", align_corners = True)
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size = 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class SegFormerDecoder(nn.Module):
    def __init__(self, outchannels, widths, scalefactors):
        super().__init__()
        self.blocks = nn.ModuleList([
            SegFormerDecoderBlock(
                inchannels = inchannels,
                outchannels = outchannels,
                scalefactor = scalefactor
            ) for inchannels, scalefactor in zip(widths, scalefactors)
        ])

    def forward(self, xs):
        outs = []
        for x, block in zip(xs, self.blocks):
            outs.append(block(x))
        return outs
    
class SegFormerBinarySegmentationHead(nn.Module):
    def __init__(self, channels, numfeatures):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels*numfeatures, channels, kernel_size = 1),
            nn.GELU(),
            nn.BatchNorm2d(channels)
        )
        self.classifier = nn.Conv2d(channels, 1, kernel_size = 1)

    def forward(self, xs):
        x = torch.cat(xs, dim = 1)
        x = self.conv(x)
        x = self.classifier(x)
        return x
    
class ConditionedSegFormer(nn.Module):
    def __init__(
        self, 
        inchannels: int,
        widths: List[int],
        depths: List[int],
        condition_dim: int,
        patch_embedding_dim: int,
        patch_sizes: List[int],
        overlap_sizes: List[int],
        num_heads: List[int],
        expansions: List[int],
        decoder_channels: int,
        scalefactors: List[int]
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            inchannels = inchannels,
            widths = widths,
            depths = depths,
            condition_dim = condition_dim,
            patch_embedding_dim = patch_embedding_dim,
            patch_sizes = patch_sizes,
            overlap_sizes = overlap_sizes,
            num_heads = num_heads,
            expansions = expansions
        )
        self.decoder = SegFormerDecoder(
            outchannels = decoder_channels,
            widths = widths[::-1],
            scalefactors = scalefactors
        )

        self.head = SegFormerBinarySegmentationHead(
            channels = decoder_channels,
            numfeatures = len(widths)
        )

    def forward(self, x, pe, condition):
        xs = self.encoder(x, pe, condition)
        xs = self.decoder(xs[::-1])
        out = self.head(xs)
        # return torch.sigmoid(out)
        return out



        