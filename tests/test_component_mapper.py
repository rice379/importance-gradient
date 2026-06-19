from importance_gradient.component_mapper import parse_transformer_component


def test_parse_opt_attention_and_mlp_components():
    assert parse_transformer_component("model.decoder.layers.0.self_attn.q_proj.weight") == (0, "Q")
    assert parse_transformer_component("model.decoder.layers.1.self_attn.k_proj.weight") == (1, "K")
    assert parse_transformer_component("model.decoder.layers.2.self_attn.v_proj.weight") == (2, "V")
    assert parse_transformer_component("model.decoder.layers.3.self_attn.out_proj.weight") == (3, "O")
    assert parse_transformer_component("model.decoder.layers.4.fc1.weight") == (4, "FC1")
    assert parse_transformer_component("model.decoder.layers.5.fc2.weight") == (5, "FC2")


def test_parse_non_component_returns_none():
    assert parse_transformer_component("model.decoder.embed_tokens.weight") is None
