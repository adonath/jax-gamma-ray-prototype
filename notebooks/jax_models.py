import dataclasses
from enum import Enum
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import fftconvolve


class register_dataclass_jax:
    """Decorator to register a dataclass with JAX."""

    def __init__(self, data_fields=None, meta_fields=None):
        self.data_fields = data_fields or []
        self.meta_fields = meta_fields or []

    def __call__(self, cls):

        jax.tree_util.register_dataclass(
            cls,
            data_fields=self.data_fields,
            meta_fields=self.meta_fields,
        )
        return cls


@register_dataclass_jax(["value"], ["unit", "frozen"])
@dataclasses.dataclass
class Parameter:
    """A model parameter.

    Attributes
    ----------
    value : jnp.ndarray
        The parameter value
    unit : str
        The unit of the parameter
    frozen : bool
        Whether the parameter is frozen
    """

    value: jnp.ndarray
    unit: str
    frozen: bool = False

    @classmethod
    def as_default(cls, **kwargs):
        """Create a default parameter."""
        return dataclasses.field(default_factory=lambda: cls(**kwargs))


class Model:
    """Model base class."""

    pass


class EnergyType(str, Enum):
    """Energy type."""

    energy = "energy"
    energy_true = "energy_true"


@register_dataclass_jax(meta_fields=["x", "y", "energy", "energy_type"])
@dataclasses.dataclass(frozen=True)
class Coords:
    """Coordinate array

    Attributes
    ----------
    x : jnp.ndarray
        The x coordinate
    y : jnp.ndarray
        The y coordinate
    energy : jnp.ndarray
        The energy coordinate
    """

    x: jnp.ndarray
    y: jnp.ndarray
    energy: jnp.ndarray
    energy_type: EnergyType = EnergyType.energy_true

    @property
    def energy_min(self):
        """The minimum true energy for each bin."""
        return self.energy[:-1]

    @property
    def energy_max(self):
        """The maximum true energy for each bin."""
        return self.energy[1:]

    @property
    def energy_log_center(self):
        """The energy log center"""
        return jnp.sqrt(self.energy_min * self.energy_max)

    @classmethod
    def from_gp_map(cls, gp_map, x_range=None, y_range=None, energy_type="energy_true"):
        """Create from a Gammapy exposure map

        Parameters
        ----------
        gp_map : `Map`
            Gammapy map
        x_range : tuple
            Range of the x coordinate
        y_range : tuple
            Range of the y coordinate
        energy_type : {"energy_true", "energy"}
            Energy type

        Returns
        -------
        coords : `Coords`
            Coordinate arrays
        """
        energy = jnp.array(
            gp_map.geom.axes[EnergyType(energy_type)].edges.to_value("TeV")
        )

        x_range = x_range or (0, gp_map.data.shape[2])
        y_range = y_range or (0, gp_map.data.shape[1])

        x, y = jnp.meshgrid(
            jnp.arange(*x_range, dtype=float), jnp.arange(*y_range, dtype=float)
        )

        return cls(x=x, y=y, energy=energy[:, None, None])


@register_dataclass_jax(["index", "reference"])
@dataclasses.dataclass
class PowerLaw(Model):
    """Power law model for the energy spectrum"""

    index: Parameter = Parameter.as_default(value=jnp.array(2.0), unit="")
    reference: Parameter = Parameter.as_default(value=jnp.array(1.0), unit="TeV")

    @staticmethod
    def evaluate(energy, index, energy_0):
        """Evaluate the power law model."""
        return (energy / energy_0) ** index

    @staticmethod
    def integrate(energy_min, energy_max, index, reference):
        """Integrate the power law model."""
        val = -1 * index + 1

        prefactor = reference / val
        upper = jnp.power((energy_max / reference), val)
        lower = jnp.power((energy_min / reference), val)
        integral = prefactor * (upper - lower)

        mask = jnp.isclose(val * prefactor, 0)

        integral = jnp.where(
            mask, reference * jnp.log(energy_max / energy_min), integral
        )

        return integral

    def call_integrate(self, coords):
        return self.integrate(
            coords.energy_min,
            coords.energy_max,
            self.index.value,
            self.reference.value,
        )

    def __call__(self, coords):
        return self.evaluate(
            coords.energy_log_center,
            self.index.value,
            self.reference.value,
        )

    @classmethod
    def as_default(cls, **kwargs):
        """Create a default parameter."""
        return dataclasses.field(default_factory=lambda: cls(**kwargs))


@register_dataclass_jax(["x_0", "y_0"], ["margin"])
@dataclasses.dataclass
class PointSource(Model):
    """Point source model"""

    x_0: Parameter = Parameter.as_default(value=jnp.array(0.0), unit="pix")
    y_0: Parameter = Parameter.as_default(value=jnp.array(0.0), unit="pix")
    margin: int = 50

    @property
    def offset(self):
        """Offset for jax dynamic slice"""
        return (
            (self.y_0.value - self.margin).astype(int),
            (self.x_0.value - self.margin).astype(int),
        )

    @property
    def size(self):
        """Size for jax dynamic slice"""
        return (2 * self.margin, 2 * self.margin)

    @property
    def x_range(self):
        """Range of the x coordinate"""
        return (
            (self.x_0.value - self.margin).astype(int),
            (self.x_0.value + self.margin).astype(int),
        )

    @property
    def y_range(self):
        """Range of the y coordinate"""
        return (
            (self.y_0.value - self.margin).astype(int),
            (self.y_0.value + self.margin).astype(int),
        )

    @property
    def cutout_slice(self):
        """Slice for the cutout"""
        return (slice(*self.y_range), slice(*self.x_range))

    @staticmethod
    def evaluate(x, y, x_0, y_0):
        """Evaluate the point source model."""
        dx = jnp.abs(x - x_0)
        dx = jnp.where(dx < 1, 1 - dx, 0)

        dy = jnp.abs(y - y_0)
        dy = jnp.where(dy < 1, 1 - dy, 0)
        return dx * dy

    def __call__(self, coords):
        return self.evaluate(coords.x, coords.y, self.x_0.value, self.y_0.value)

    @classmethod
    def as_default(cls, **kwargs):
        """Create a default parameter."""
        return dataclasses.field(default_factory=lambda: cls(**kwargs))


@register_dataclass_jax(["amplitude", "spectral", "spatial"])
@dataclasses.dataclass
class FluxModel(Model):
    """Sky model"""

    amplitude: Parameter = Parameter.as_default(
        value=jnp.array(1e-10), unit="TeV^-1 m^-2 s^-1"
    )
    spectral: PowerLaw = PowerLaw.as_default()
    spatial: PointSource = PointSource.as_default()

    def __call__(self, coords):
        return (
            self.amplitude.value
            * self.spectral.call_integrate(coords)
            * self.spatial(coords)
        )

    @property
    def cutout_slice(self):
        """Cutout slice for the spatial model."""
        return (...,) + self.spatial.cutout_slice

    @property
    def offset(self):
        """Offset for jax dynamic slice"""
        return (0,) + self.spatial.offset


@register_dataclass_jax(["norm", "spectral"])
@dataclasses.dataclass
class NormModel(Model):
    """Norm model"""

    norm: Parameter = Parameter.as_default(value=jnp.array(1), unit="")
    spectral: PowerLaw = PowerLaw.as_default(
        index=Parameter(value=jnp.array(0), unit="")
    )

    def __call__(self, coords):
        return self.norm.value * self.spectral(coords)

    @property
    def cutout_slice(self):
        """Cutout slice for the spatial model."""
        return (..., ..., ...)

    @property
    def offset(self):
        """Offset for jax dynamic slice"""
        return (0, 0, 0)


@register_dataclass_jax(["model"], ["exposure", "coords_true", "psf", "edisp"])
@dataclasses.dataclass
class NPredSourceModel:
    """Npred source model"""

    model: FluxModel
    exposure: jnp.ndarray
    coords_true: Coords
    psf: Optional[jnp.ndarray] = None
    edisp: Optional[jnp.ndarray] = None

    @classmethod
    def from_gp_dataset(cls, dataset, model):
        """Create from a Gammapy dataset"""
        # TODO: extract at model position
        edisp_gp = dataset.edisp.get_edisp_kernel()

        # TODO: extract at model position
        psf = dataset.psf.get_psf_kernel(geom=dataset.exposure.geom)

        coords_true = Coords.from_gp_map(
            dataset.exposure,
            x_range=model.spatial.x_range,
            y_range=model.spatial.y_range,
            energy_type="energy_true",
        )

        return cls(
            model=model,
            exposure=jnp.array(dataset.exposure.data[model.cutout_slice]),
            coords_true=coords_true,
            edisp=jnp.array(edisp_gp.data),
            psf=jnp.array(psf.psf_kernel_map.data),
        )

    @property
    def flux(self):
        """Compute flux"""
        return self.model(self.coords_true)

    @property
    def needs_update(self):
        pass

    def npred(self):
        """Compute npred"""
        npred = self.flux * self.exposure

        if self.psf is not None:
            npred = fftconvolve(npred, self.psf, mode="same", axes=(-2, -1))

        if self.edisp is not None:
            npred = jnp.matmul(npred.T, self.edisp).T

        return npred


@register_dataclass_jax(["model"], ["data", "coords"])
@dataclasses.dataclass(frozen=True)
class NPredTemplateModel:
    """Data template model"""

    model: NormModel
    data: jnp.ndarray
    coords: Coords

    def npred(self):
        """Compute npred"""
        correction = self.model(self.coords)
        return correction * self.data

    @classmethod
    def from_gp_dataset(cls, dataset, model):
        """Create from a Gammapy dataset"""

        coords = Coords.from_gp_map(
            dataset.background,
            energy_type="energy",
        )

        return cls(
            model=model,
            data=jnp.array(dataset.background.data),
            coords=coords,
        )


NPredModel = Union[NPredSourceModel, NPredTemplateModel]

NPRED_MODEL_CLS = {FluxModel: NPredSourceModel, NormModel: NPredTemplateModel}


@register_dataclass_jax(["models"], ["shape"])
@dataclasses.dataclass(frozen=True)
class NPredModels:
    """Collection of NPredModels."""

    models: dict[str, NPredModel]
    shape: tuple[int, int, int]

    @classmethod
    def from_gp_dataset(cls, dataset, models, names=None):
        """Create from a Gammapy dataset"""
        npred_models = {}

        if names is None:
            names = [f"model-{idx}" for idx in range(len(models))]

        for name, model in zip(names, models):
            npred_model = NPRED_MODEL_CLS[type(model)].from_gp_dataset(dataset, model)
            npred_models[name] = npred_model

        shape = dataset.counts.data.shape
        return cls(models=npred_models, shape=shape)

    def npred(self):
        """Compute npred"""
        npred = jnp.zeros(self.shape)

        for _, npred_model in self.models.items():
            npred_source = npred_model.npred()

            offset = npred_model.model.offset

            npred_old = lax.dynamic_slice(npred, offset, npred_source.shape)

            npred = lax.dynamic_update_slice(npred, npred_source + npred_old, offset)

        return npred
