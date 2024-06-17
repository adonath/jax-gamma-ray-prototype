import getpass
import importlib
import logging
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from benchmark import read_config
from psrecord.main import monitor
from ruamel.yaml import YAML

log = logging.getLogger(__name__)

THIS_REPO = Path(__file__).parent
MONITOR_OPTIONS = {"duration": None, "interval": 0.5, "include_children": True}

CSV_FORMAT = {
    "sep": "\t",
    "float_format": "%12.3f",
    "index": False,
}


def image_size_to_str(image_size):
    """Image size to str"""
    return f"{image_size[0]}x{image_size[1]}"


def get_provenance_folder_name(info):
    """Get provenance folder name"""
    env = info["env"]

    gpu = env.get("gpu", "")
    name = f"{env['user']}-{env['cpu']}-{env['system']}-{env['machine']}"

    if gpu:
        name += gpu

    return name


def get_provenance():
    """Compute provenance info about software and data used."""
    data = {
        "env": {
            "user": getpass.getuser(),
            "machine": platform.machine(),
            "system": platform.system(),
            "cpu": platform.processor(),
        },
        "software": {},
    }

    data["software"]["python_executable"] = sys.executable
    data["software"]["python_version"] = platform.python_version()
    data["software"]["jax"] = str(importlib.import_module("jax").__version__)
    data["software"]["numpy"] = str(importlib.import_module("numpy").__version__)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        data["env"]["gpu"] = device_name

    return data


@click.group()
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    """
    Run and manage Jolideco benchmarks.
    """
    levels = dict(
        debug=logging.DEBUG,
        info=logging.INFO,
        warning=logging.WARNING,
        error=logging.ERROR,
        critical=logging.CRITICAL,
    )
    logging.basicConfig(level=levels[log_level])
    log.setLevel(level=levels[log_level])

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-benchmark", help="Run Jolideco benchmarks")
@click.option(
    "--filename", type=str, help="Benchmark configuration filename", default="all"
)
@click.option("--image-size", type=int, help="Image size", default=-1)
@click.option("--use-gpu", is_flag=True, help="Use GPU?")
def run_benchmarks(filename, image_size, use_gpu):

    if filename == "all":
        filenames = (THIS_REPO / "benchmarks").glob("*.yaml")
    else:
        filenames = [THIS_REPO / filename]

    if image_size == -1:
        image_sizes = IMAGE_SIZES
    else:
        idx = int(np.log2(image_size) - 5)
        image_sizes = IMAGE_SIZES[slice(idx, idx + 1)]

    info = get_provenance()

    for filename in filenames:
        name = read_config(filename=filename)["name"]
        results_folder = (
            THIS_REPO / f"results/{name}" / get_provenance_folder_name(info=info)
        )
        results_folder.mkdir(exist_ok=True, parents=True)

        provenance_filename = results_folder / "provenance.yaml"

        with provenance_filename.open("w") as fh:
            yaml = YAML()
            yaml.default_flow_style = False
            log.info("Writing {}".format(provenance_filename))
            yaml.dump(info, fh)

        for image_size in image_sizes:
            results_folder_run = results_folder / image_size_to_str(
                image_size=image_size
            )
            results_folder_run.mkdir(exist_ok=True)
            results_filename = results_folder_run / "results.txt"
            plot_filename = results_folder_run / "results.png"

            run_single_benchmark(
                filename=filename,
                image_size=image_size,
                logfile=str(results_filename),
                plot=str(plot_filename),
                use_gpu=use_gpu,
                **MONITOR_OPTIONS,
            )

            log.info("Writing {}".format(results_filename))
            log.info("Writing {}".format(plot_filename))


def run_single_benchmark(filename, image_size, use_gpu, **kwargs):
    script_path = (THIS_REPO / "benchmark.py").absolute()

    cmd = [
        "python",
        str(script_path),
        "--filename",
        str(filename),
        "--image_size",
        str(image_size[0]),
        str(image_size[1]),
    ]

    if use_gpu:
        cmd = cmd + ["--use-gpu"]

    log.info(f"Executing command: {' '.join(cmd)}")

    with tempfile.TemporaryDirectory() as path:
        process = subprocess.Popen(cmd, cwd=str(path))
        monitor(process.pid, **kwargs)


@cli.command("gather-results", help="Run Jolideco benchmarks")
@click.option(
    "--filename", type=str, help="Benchmark configuration filename", default="all"
)
def gather_results(filename):
    if filename == "all":
        filenames = (THIS_REPO / "benchmarks").glob("*.yaml")
    else:
        filenames = [THIS_REPO / filename]

    for filename in filenames:
        name = read_config(filename=filename)["name"]
        results_folder = THIS_REPO / f"results/{name}"

        subdirs = [path for path in results_folder.iterdir() if path.is_dir()]

        columns = ["image-size"] + [str(_.parts[-1]) for _ in subdirs]
        time_total = pd.DataFrame(columns=columns)
        time_total["image-size"] = [float(_[0]) for _ in IMAGE_SIZES]

        for subdir in subdirs:
            results = pd.DataFrame(
                columns=[
                    "image-size",
                    "total-time",
                    "cpu-max",
                    "cpu-mean",
                    "memory-max",
                ]
            )

            for idx, image_size in enumerate(IMAGE_SIZES):
                results_filename = (
                    subdir / image_size_to_str(image_size=image_size) / "results.txt"
                )
                log.info(f"Reading {results_filename}")

                t, cpu, _, memory = np.loadtxt(results_filename, unpack=True)
                row = [
                    int(image_size[0]),
                    max(t),
                    max(cpu[2:]),
                    np.mean(cpu[2:]),
                    np.max(memory),
                ]
                results.loc[idx] = row

            filename = subdir / "summary.txt"

            log.info(f"Writing {filename}")

            header = ["{:>12s}".format(_) for _ in results.columns]
            with filename.open("w") as f:
                results.to_csv(f, header=header, **CSV_FORMAT)

            time_total[str(subdir.parts[-1])] = results["total-time"]

        filename = results_folder / "summary-time.txt"
        log.info(f"Writing {filename}")
        header = ["{:>24s}".format(_) for _ in time_total.columns]

        with filename.open("w") as f:
            CSV_FORMAT["float_format"] = "%24.3f"
            time_total.to_csv(f, header=header, **CSV_FORMAT)


@cli.command("plot-results", help="Plot Jolideco benchmark results")
@click.option(
    "--filename", type=str, help="Benchmark configuration filename", default="all"
)
def plot_results(filename):
    if filename == "all":
        filenames = (THIS_REPO / "benchmarks").glob("*.yaml")
    else:
        filenames = [THIS_REPO / filename]

    for filename in filenames:
        name = read_config(filename=filename)["name"]
        results_folder = THIS_REPO / f"results/{name}"

        df = pd.read_csv(
            results_folder / "summary-time.txt", sep="\t", skipinitialspace=True
        )

        ax = df.plot(
            x="image-size",
            y=None,
            loglog=True,
            xlabel="Image size / pix",
            ylabel="Time / s",
            ylim=(1, 10_000),
            style="-o",
            figsize=(10, 6.25),
        )

        alpha = 0.15
        ypos = [6, 60, 600]
        ax.hlines(y=ypos, xmin=1, xmax=2000, color="k", alpha=alpha, lw=0.5)
        ax.set_xlim(28, 1500)

        ax.text(x=35, y=1.2 * ypos[0], s="10 min. / 1000 epochs", alpha=alpha)
        ax.text(x=35, y=1.2 * ypos[1], s="100 min. / 1000 epochs", alpha=alpha)
        ax.text(x=35, y=1.2 * ypos[2], s="1000 min. / 1000 epochs", alpha=alpha)

        filename = results_folder / "summary-time.png"
        log.info(f"Writing {filename}")
        plt.savefig(filename, dpi=120)


if __name__ == "__main__":
    cli()
