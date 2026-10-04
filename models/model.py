import torch
import torch.nn as nn
import timm
from .head import GatedCrossAttentionPooling

class RSNAKADModel(nn.Module):
    def __init__(
        self,
        num_classes,
        encoder_name: str = "convnext_tiny.fb_in1k",
        pretrained: bool = True,
        n_queries: int = 16,
        num_heads: int = 8,
        pooling_inner_dim: int = 128, 
    ):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )

        self.d_model = int(self.encoder.num_features)
        self.norm = nn.BatchNorm2d(self.d_model)

        self.cap = GatedCrossAttentionPooling(
            d_model=self.d_model,
            n_queries=n_queries,
            num_heads=num_heads,
            gate_type="headwise",
            inner_dim=pooling_inner_dim  
        )
    
        self.classifier = nn.Linear(n_queries * pooling_inner_dim if pooling_inner_dim else n_queries * self.d_model, self.num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.logits(feats)

    def features(self, x):
        B, N_slices, C, H, W = x.shape
        x_flat = x.reshape(B * N_slices, C, H, W)

        feats = self.encoder(x_flat)
        feats = self.norm(feats)
        
        Cf, Hf, Wf = feats.shape[1:]
        
        return feats.view(B, N_slices, Cf, Hf, Wf).permute(0, 2, 1, 3, 4).contiguous()

    def logits(self, x):
        pooled = self.cap(x)
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    model = RSNAKADModel(12)
    input = torch.randn((4, 32, 1, 256, 256))
    out = model(input)
    print(out.shape)

    feats = model.features(input)
    print(feats.shape)