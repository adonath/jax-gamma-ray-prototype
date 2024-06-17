import timeit

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy import units as u
from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
)
from gammapy.modeling.models import SkyModel as GPSkyModel
from jax_models import FLUX_FACTOR, FluxModel, NormModel, PointSource, PowerLaw

from notebooks.jax_models import NPredModels

jax.config.update("jax_enable_x64", True)


RANDOM_STATE = np.random.RandomState(982374)


N_SOURCES = 2 ** np.arange(7)
DATASET_SIZES = 2 ** np.arange(8, 11)
N_REPEATS = 10
MARGIN = 69


def profile_func(func):
    timer = timeit.Timer("func()", globals={"func": func})
    return timer.timeit(N_REPEATS)


def get_gp_model(dataset, n_sources=1):
    models = Models()
    models.append(FoVBackgroundModel(dataset_name=dataset.name))

    shape = dataset.counts.data.shape

    positions = RANDOM_STATE.uniform(MARGIN, shape[0] - MARGIN, size=(n_sources, 2))
    positions = dataset.counts.geom.pix_to_coord(positions)

    for idx in range(n_sources):
        x_0, y_0 = positions[idx]
        point = PointSpatialModel(x_0=x_0 * u.deg, y_0=y_0 * u.deg, frame="galactic")
        spectral = PowerLawSpectralModel(amplitude="1e-10 cm-2 s-1 TeV-1")

        model = GPSkyModel(
            spatial_model=point, spectral_model=spectral, name=f"src-{idx}"
        )
        models.append(model)

    return models


def get_jax_model(dataset, n_sources=1):
    models = []

    bkg_jax = NormModel()
    bkg_jax.spectral.index.value = jnp.array(0.0)

    models.append(bkg_jax)

    models.append(FluxModel())
    models.append(NormModel())

    shape = dataset.counts.data.shape

    positions = RANDOM_STATE.uniform(MARGIN, shape[0] - MARGIN, size=(n_sources, 2))

    for idx in range(n_sources):
        point = PointSource()
        point.x_0.value = jnp.array(positions[idx, 0])
        point.y_0.value = jnp.array(positions[idx, 1])

        source_jax = FluxModel(spectral=PowerLaw(), spatial=point)
        source_jax.amplitude.value = jnp.array(1e-6) / FLUX_FACTOR

        models.append(source_jax)

    return models


def profile_n_sources_jax(dataset, use_jit=False):

    results = []

    for n_sources in N_SOURCES:

        npred_model_jax = NPredModels.from_gp_dataset(
            dataset, get_jax_model(dataset, n_sources=n_sources)
        )

        func = npred_model_jax.npred

        if use_jit:
            func = jax.jit(func)

        result = profile_func(func)
        results.append((n_sources, result))

    return results


def profile_n_sources_gp(dataset):
    results = []

    for n_sources in N_SOURCES:

        dataset.models = get_gp_model(dataset, n_sources=n_sources)
        result = profile_func(dataset.npred)
        results.append((n_sources, result))

    return results


def profile_n_sources(dataset):
    data = {}

    data["n_sources"] = N_SOURCES

    data["gp"] = profile_n_sources_gp(dataset)
    data["jax"] = profile_n_sources_jax(dataset, use_jit=False)
    data["jax-jit"] = profile_n_sources_jax(dataset, use_jit=True)

    return pd.DataFrame(data=data)


def profile_dataset_sizes(func, n_sources=1):
    pass


if __name__ == "__main__":
    dataset = MapDataset.read("../data/test-dataset-0.fits")
    result = profile_n_sources(dataset)
    result.to_csv("results/n_sources.csv", index=False)
