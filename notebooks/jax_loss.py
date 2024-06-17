import dataclasses
from typing import Optional

from jax import numpy as jnp
from jax_models import NPredModels, register_dataclass_jax

TINY_VALUE = 1e-25


@register_dataclass_jax(["npred_models"], ["counts", "mask"])
@dataclasses.dataclass
class CashFitStatistic:
    """Cash fit statistic for a set of observed counts and NPredModels.

    Attributes
    ----------
    counts : jnp.array
        Observed counts
    npred_models : NPredModels
        Predicted counts
    weights : jnp.array
        Weights for the fit statistic
    """

    counts: jnp.array
    npred_models: NPredModels
    weights: Optional[jnp.array] = None

    @staticmethod
    def evaluate(counts, npred):
        """Evaluate the Cash fit statistic.

        Parameters
        ----------
        counts : jnp.array
            Observed counts
        npred : jnp.array
            Predicted counts

        Returns
        -------
        jnp.array
            Cash fit statistic
        """
        npred = jnp.maximum(npred, TINY_VALUE)
        return npred - counts * jnp.log(npred)

    def __call__(self):
        """Evaluate the Cash fit statistic."""
        npred = self.npred_models.npred()
        return 2 * jnp.sum(self.evaluate(self.counts, npred) * self.weights)

    @classmethod
    def from_gp_dataset(cls, dataset, models):
        """Create a CashFitStatistic from a GP dataset.

        Parameters
        ----------
        dataset : MapDataset
            Dataset
        models : list[Model]
            Models

        Returns
        -------
        CashFitStatistic
            Cash fit statistic
        """
        npred_models = NPredModels.from_gp_dataset(dataset, models)

        return cls(
            counts=jnp.array(dataset.counts.data),
            npred_models=npred_models,
            weights=jnp.array(dataset.mask_safe.data, dtype=float),
        )


@register_dataclass_jax(["fit_statistics"])
@dataclasses.dataclass
class FitStatistics:
    """List of fit statistics

    Attributes
    ----------
    fit_statistics : dict[str, CashFitStatistic]
        Dictionary of fit statistics
    """

    fit_statistics: dict[str, CashFitStatistic]

    def __call__(self):
        """Evaluate the fit statistics."""
        return jnp.array([stat() for stat in self.fit_statistics.values()]).sum()


@register_dataclass_jax()
@dataclasses.dataclass
class HomogeneousFitStatistics:
    """Homogeneous fit statistics"""

    pass
