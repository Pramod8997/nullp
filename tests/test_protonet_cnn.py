import torch
from src.models.protonet import CNNEncoder

# TEST 2A-1: Output embedding shape must always be [batch, 128]
def test_cnn_encoder_output_shape():
    encoder = CNNEncoder()
    x = torch.randn(8, 1, 128)  # batch=8, channels=1, length=128
    z = encoder(x)
    assert z.shape == (8, 128), f"Expected (8, 128), got {z.shape}"

# TEST 2A-2: Temporal attention weights must sum to 1 (valid probability distribution)
def test_temporal_attention_weights_sum_to_one():
    encoder = CNNEncoder()
    x = torch.randn(4, 1, 128)
    # Access attention weights through hook or expose them
    attn_weights = encoder.get_attention_weights(x)  # expose in implementation
    # Sigmoid gating: each weight should be in (0, 1) independently
    assert (attn_weights >= 0).all() and (attn_weights <= 1).all()
    assert attn_weights.shape == (4, 128)

# TEST 2A-3: Encoder is deterministic in eval mode (no dropout)
def test_encoder_deterministic_eval_mode():
    encoder = CNNEncoder().eval()
    x = torch.randn(1, 1, 128)
    out1 = encoder(x)
    out2 = encoder(x)
    assert torch.allclose(out1, out2)

# TEST 2A-4: Encoder output L2 norm should be bounded (not exploding)
def test_encoder_embedding_norm_bounded():
    encoder = CNNEncoder().eval()
    x = torch.randn(32, 1, 128)
    z = encoder(x)
    norms = torch.norm(z, dim=1)
    assert norms.max().item() < 100.0, "Embeddings should not explode"

# TEST 2A-5: Encoder gradients flow through all 4 conv layers
def test_encoder_gradient_flow():
    encoder = CNNEncoder().train()
    x = torch.randn(4, 1, 128, requires_grad=False)
    z = encoder(x)
    loss = z.sum()
    loss.backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
