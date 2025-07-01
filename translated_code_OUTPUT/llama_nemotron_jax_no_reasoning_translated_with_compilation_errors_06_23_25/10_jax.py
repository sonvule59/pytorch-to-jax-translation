18.5
lang: python
import jax
from jax import numpy as jnp
from allennlp.modules.span_extractors import SpanExtractor, SpanExtractorWithSpanWidthEmbedding
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import SpanExtractorWithSpanWidthEmbeding
from allennlp.nn import util

@SpanExtractor.register("endpoint")
def EndpointSpanExtractor(
    input_dim: int,
    combination: str = "x,y",
    num_width_embeddings: int = None,
    span_width_embedding_dim: int = None,
    bucket_widths: bool = False,
    use_exclusive_start_indices: bool = False,
) -> jax.nn.Module:
    # Parameters
    # Initialize parameters similar to Torch's parameters
    start_embedding: jnp.ndarray = None
    end_embedding: jnp.ndarray = None
    _combination: str = combination
    _use_exclusive_start_indices: bool = use_exclusive_start_indices
    _start_sentinel: jnp.ndarray = None  # JAX equivalent of Torch's _start_sentinel

    super().__init__(
        input_dim=input_dim,
        num_width_embeddings=num_width_embeddings,
        span_width_embedding_dim=span_width_embedding_dim,
        bucket_widths=bucket_widths,
    )

    # Save the JAX version of parameters
    self._start_embedding = start_embedding
    self._end_embedding = end_embedding
    self._combination = _combination
    self._use_exclusive_start_indices = _use_exclusive_start_indices
    self._start_sentinel = _start_sentinel

    # Define the forward pass
    def _embed_spans(
        sequence_tensor: jnp.ndarray,
        span_indices: jnp.ndarray,
        sequence_mask: jnp.ndarray = None,
        span_indices_mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        # shape (batch_size, num_spans)
        batch_size, num_spans = sequence_tensor.shape[:2]

        # Extract span start and end indices
        span_starts = span_indices[:, 0].flatten()
        span_ends = span_indices[:, 1].flatten()

        # Apply mask if provided
        if span_indices_mask is not None:
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        # Handle exclusive start indices
        if _use_exclusive_start_indices:
            # Adjust for exclusive indices by subtracting 1 from forward spans
            exclusive_span_starts = span_starts - 1
            # Create mask for sentinels (batch_size, num_spans, 1)
            sentinel_mask = (exclusive_span_starts == -1).astype("bool")[:, :, None]
            exclusive_span_starts = exclusive_span_starts * ~sentinel_mask.squeeze(-1)
            # Check for negative indices
            if (exclusive_span_starts < 0).any():
                raise ValueError("Adjusted span indices must lie within sequence tensor bounds.")

            # Get embeddings for spans (batch_size, num_spans, embedding_dim)
            start_embeddings = jax.vmap(jax.index_select, (0, -1, -2), sequence_tensor)[exclusive_span_starts]
            end_embeddings = jax.vmap(jax.index_select, (0, -1, -2), sequence_tensor)[span_ends]

            # Insert sentinel values where indices were out of bounds
            sentinel_values = jnp.full((batch_size, num_spans, self.input_dim), self._start_sentinel, dtype=jnp.float32)
            start_embeddings = jnp.where(~sentinel_mask.squeeze(-1), start_embeddings, sentinel_values)
        else:
            # Inclusive indices
            start_embeddings = jax.vmap(jax.index_select, (0, -1, -2), sequence_tensor)[span_starts]
            end_embeddings = jax.vmap(jax.index_select, (0, -1, -2), sequence_tensor)[span_ends]

        # Combine start and end embeddings according to the combination string
        combined_embeddings = util.combine_embeddings(
            self._combination, start_embeddings, end_embeddings
        )

        # Handle span width embeddings if present
        if self.num_width_embeddings is not None:
            # Assuming span_widths is a vector of shape (batch_size, num_spans)
            # Compute bin width for each span
            if bucket_widths:
                bin_widths = jnp.logspace(0, 1, num=num_width_embeddings + 1)[:-1].reshape(-1, 1)
            else:
                bin_widths = jnp.ones(num_width_embeddings)

            # Compute bin index for each span
            span_widths = jnp.max(span_ends) - jnp.min(span_starts)
            bin_indices = jnp.clip(
                (span_starts - jnp.min(span_starts)) / span_widths * num_width_embeddings,
                a=0, b=num_width_embeddings - 1