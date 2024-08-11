def forward_and_loss_fn(params,
                        *,
                        state,
                        input_tokens: jax.Array,            # Shape [B, L]
                        input_mask: jax.Array,              # Shape [B, L]
                        positions: jax.Array,               # Shape [B, L]
                        attention_mask: jax.Array,          # [B, L, L]
                        ) -> jax.Array:
  """Forward pass and loss function.

  Args:
    params: model's input parameters.
    model: gemma transformer model to call.
    input_tokens: input tokens sequence, shape [B, L].
    input_mask: tokens to ignore when computing the loss, shape [B, L].
    positions: relative position of each token, shape [B, L].
    attention_mask: input attention mask, shape [B, L].

  Returns:
    Softmax cross-entropy loss for the next-token prediction task.
  """

  # Forward pass on the input data.
  # No attention cache is needed here.
  logits, _ = state.apply_fn(
        {"params": params},
        input_tokens,
        positions,
        None,              # Attention cache is None.
        attention_mask,
    )

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1]

  # Similarly, the first token cannot be predicteds.
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels into one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Normalisation factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the nll loss.
  loss =  -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
  # pdb.set_trace()
  return loss