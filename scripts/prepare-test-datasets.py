import logging
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.data import (
    FixedPointingInfo,
    Observation,
    observatory_locations,
)
from gammapy.datasets import MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


PATH_CALDB = Path("$GAMMAPY_DATA/cta-caldb")

PATH = Path(__file__).parent

GEOM_SPEC = {
    "skydir": SkyCoord("0d", "0d", frame="galactic"),
    "width": "10.24 deg",
    "binsz": "0.01d",
    "frame": "galactic",
}

OBSERVATION_SPEC = {
    "livetime": 1 * u.hr,
    "pointing": SkyCoord("0d", "0d", frame="galactic"),
    "filename_irf": "Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz",
}


def get_observation(livetime, pointing, filename_irf, obs_id=1):
    """Create a test observation"""
    # telescope is pointing at a fixed position in ICRS for the observation
    pointing = FixedPointingInfo(fixed_icrs=pointing.icrs)
    location = observatory_locations["cta_south"]

    irfs = load_irf_dict_from_file(PATH_CALDB / filename_irf)

    return Observation.create(
        obs_id=obs_id,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=location,
    )


def get_dataset_spec():
    """Create a test geometry"""
    energy = MapAxis.from_energy_bounds("30 GeV", "300 TeV", nbin=10, per_decade=True)

    energy_axis_true = MapAxis.from_energy_bounds(
        "30 GeV", "300 TeV", nbin=20, per_decade=True, name="energy_true"
    )

    geom = WcsGeom.create(**GEOM_SPEC, axes=[energy])

    return {
        "geom": geom,
        "energy_axis_true": energy_axis_true,
    }


def get_dataset(spec, observation):
    """Get dataset with a given geometry and observation."""
    log.info(f"Creating dataset for observation {observation.obs_id}")
    empty = MapDataset.create(**spec, name="dataset")
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, observation)
    return dataset


if __name__ == "__main__":
    spec = get_dataset_spec()

    observation = get_observation(**OBSERVATION_SPEC, obs_id=1)
    dataset = get_dataset(spec, observation)

    path = PATH / "../data"
    filename = path / "test-dataset.fits"
    log.info(f"Writing {filename}")
    dataset.write(filename, overwrite=True)
