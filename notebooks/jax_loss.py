import dataclasses
from typing import Optional

import jax
from jax import numpy as jnp
from jax_models import NPredModels, register_dataclass_jax

TINY_VALUE = 1e-25


@register_dataclass_jax(["npred_models"], ["counts", "mask"])
@dataclasses.dataclass
class CashFitStatistic:
    """Cash fit statistic for a set of observed counts and NPredModels."""

    counts: jnp.array
    npred_models: NPredModels
    mask: Optional[jnp.array] = None

    @staticmethod
    def evaluate(counts, npred):
        """Evaluate the Cash fit statistic."""
        npred = jnp.maximum(npred, TINY_VALUE)
        return npred - counts * jnp.log(npred)

    def __call__(self):
        """Evaluate the Cash fit statistic."""
        npred = self.npred_models.npred()
        return 2 * jnp.sum(self.evaluate(self.counts, npred) * self.mask)

    @classmethod
    def from_gp_dataset(cls, dataset, models):
        """Create a CashFitStatistic from a GP dataset."""
        npred_models = NPredModels.from_gp_dataset(dataset, models)

        return cls(
            counts=jnp.array(dataset.counts.data),
            npred_models=npred_models,
            mask=jnp.array(dataset.mask_safe.data, dtype=float),
        )
