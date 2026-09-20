import torch
from torch import nn


# --- THE ANOBERT CONTAINER ---
class AnoBertEmbeddingLayer(nn.Module):
    def __init__(
        self,
        num_continuous_features: int = 13,
        num_categorical_features: int = 2,
        feature_dim: int = 51,
        d_model: int = 32,
        max_seq_len: int = 34,
        # 65537 = the old raw-port scheme's default (0-65535 + 1 for the mask/pad
        # sentinel), kept as the default here only so this class stays constructible
        # standalone (e.g. in tests) without wiring through the quantized vocab size.
        # Real runs pass this in from schema.port_vocab_size(params.port_top_k) + 1
        # (see training/trainer_handler.py) so it matches whatever the preprocessing
        # stage's port quantization produced.
        num_port_classes: int = 65537,
    ):
        super().__init__()
        self.num_continuous_features = num_continuous_features
        self.num_categorical_features = num_categorical_features
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_port_classes = num_port_classes

        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

        # Independent Continuous Feature Extractors (Projecting 1D inputs to feature_dim space)
        self.numerical_projectors = nn.ModuleList(
            [
                nn.Linear(1, self.feature_dim, bias=False)
                for _ in range(self.num_continuous_features)
            ]
        )
        # Discrete Token Embeddings (Allocating unique dictionary rows for masking tokens)
        self.port_dst_embedding = nn.Embedding(num_port_classes, self.feature_dim)
        self.proto_embedding = nn.Embedding(257, self.feature_dim)  # 0-255   + 1 for the mask

        # Mathematically derived bottleneck fusion layer bounds
        total_concat_dim = (
            self.num_continuous_features + self.num_categorical_features
        ) * self.feature_dim
        self.proj_layer = nn.Linear(total_concat_dim, d_model)
        # Explicitly stable initialization parameters for compression scaling boundaries
        nn.init.xavier_uniform_(self.proj_layer.weight)
        if (
            self.proj_layer.bias is not None
        ):  # setting biases to 0 initially is good practice to start with symmetric distributions
            self.proj_layer.bias.data.fill_(0)

        # Swapped from legacy ReLU to modern GELU to prevent vanishing gradients
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x_continuous: torch.Tensor,  # expected shape [Batch x Seq_Len x 13 ]
        x_proto: torch.Tensor,  # expected shape [Batch x Seq_Len ]
        x_port: torch.Tensor,  # expected shape [Batch x Seq_Len ]
    ) -> torch.Tensor:

        B, S, F = x_continuous.shape  # B=Batch, S=Seq_Len, F=Continuous Features (13)
        device = x_continuous.device

        projected_features = []
        for i in range(F):
            numerical_feat = x_continuous[:, :, [i]]  # Shape [B, S, 1]
            projected_slice = self.numerical_projectors[i](numerical_feat)
            projected_features.append(projected_slice)

        projected_features.append(self.port_dst_embedding(x_port))

        projected_features.append(self.proto_embedding(x_proto))

        # 13 numerical + 2 categorical tensors = 15 tensors of shape [B, S, self.feature_dim (51)]
        # Concatenating along dim=2 multiplies the feature dimension by 15, thus fused_features has dim [B, S, 765]
        fused_features = torch.cat(projected_features, dim=2)
        # Compress the 765-dimensional features to down to model's d_model size (32 todo increase later)
        embed_tokens = self.activation(self.proj_layer(fused_features))

        # todo replace by something more sophisticated RoPE or sine/cosine oder so
        # Generate position integers [0, 1, 2, ..., S-1]
        position_ids = torch.arange(S, dtype=torch.long, device=device)
        # Expand across the batch dimension to match the input grid
        position_ids = position_ids.unsqueeze(0).expand(B, S)  # Shape: [B, S]
        # Look up the position embeddings
        position_embeds = self.position_embeddings(position_ids)  # Shape: [B, S, 32])

        # Sum tokens and positions together for the final transformer sequence input
        return self.dropout(embed_tokens + position_embeds)  # Shape: [B, S, 32]


# --- 3. THE ANOBERT CONTAINER ---
class AnoBert(nn.Module):
    def __init__(
        self,
        num_continuous_features: int = 13,
        num_categorical_features: int = 2,
        feature_dim: int = 51,
        input_dim: int = 15,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 34,
        num_port_classes: int = 65537,  # see AnoBertEmbeddingLayer's docstring on this default
    ):
        super().__init__()
        self.num_continuous_features = num_continuous_features
        self.num_categorical_features = num_categorical_features
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_port_classes = num_port_classes

        # initialize the custom embedding layer using the parameters from constructor
        self.embed = AnoBertEmbeddingLayer(
            num_continuous_features=self.num_continuous_features,
            num_categorical_features=num_categorical_features,
            feature_dim=self.feature_dim,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            num_port_classes=num_port_classes,
        )

        # standard transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,  # why 4?
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.num_layers
        )

        # multi-task outputheads
        self.reconstruction_continuous = nn.Linear(
            self.d_model, self.num_continuous_features
        )  # Output size: 13
        # Logits for num_port_classes-1 quantized port classes + the mask/pad sentinel
        self.reconstruction_port = nn.Linear(self.d_model, num_port_classes)
        self.reconstruction_proto = nn.Linear(self.d_model, 257)  # Logits for 256 protocols + Mask

    def forward(
        self,
        x_continuous: torch.Tensor,  # shape [B, S, self.num_continous_features (13)],
        x_proto: torch.Tensor,  # shape [B, S],
        x_port: torch.Tensor,  # shape [B, S],
        key_padding_mask: torch.Tensor | None = None,  # shape [B, S]; True = pad/ignore
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # C = Continuous Features (13)
        num_continuous = x_continuous.shape[2]
        if num_continuous != self.num_continuous_features:
            raise ValueError(
                "Input has "
                f"{num_continuous} continuous features, but the model expects "
                f"{self.num_continuous_features}."
            )
        # 1. Fuse heterogeneous streams into abstract transformer sequence tokens
        x = self.embed(x_continuous, x_proto, x_port)

        # 2. Extract contextual multi-flow representations via deep attention
        context_vectors = self.transformer(
            x, src_key_padding_mask=key_padding_mask
        )  # Output Shape: [B, S, 32]

        # 3. Pure mathematical mapping out to specialized target distributions
        out_continuous = self.reconstruction_continuous(context_vectors)  # Shape: [B, S, 13]
        out_proto = self.reconstruction_proto(context_vectors)  # Shape: [B, S, 257]
        out_port = self.reconstruction_port(context_vectors)  # Shape: [B, S, 65537]

        return out_continuous, out_proto, out_port
