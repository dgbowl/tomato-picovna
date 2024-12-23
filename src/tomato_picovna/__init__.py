from typing import Any, Optional
from types import ModuleType
from tomato.driverinterface_1_0 import ModelInterface, Attr, Task, Reply, in_devmap
from pathlib import Path
import psutil
import sys
import importlib
from time import perf_counter, sleep
from pydantic import BaseModel, model_validator
import numpy as np
import logging
from datetime import datetime
from xarray import Dataset

vna: ModuleType = None

BANDWIDTH_SET = {10, 50, 100, 500, 1_000, 5_000, 10_000, 35_000, 70_000, 140_000}
POINTS_SET = {11, 51, 101, 201, 401, 801, 1001, 2001, 3001, 4001, 5001, 6001, 7001}
logger = logging.getLogger(__name__)


def estimate_sweep_time(bw: int, npoints: int):
    # Parameters obtained using a curve fit to various npoints & bandwidths
    c0, c1, c2 = [7.17562593e-01, 1.83389054e00, 1.28624374e-04]
    return c0 + npoints * (c1 / bw + c2)


class Sweep(BaseModel):
    start: float
    stop: float
    points: Optional[int] = None
    step: Optional[float] = None

    @model_validator(mode="after")
    def check_points_or_step(self):
        if self.points is None and self.step is None:
            raise ValueError("Must supply either 'points' or 'step'.")
        if self.points is not None and self.step is not None:
            raise ValueError("Must supply either 'points' or 'step', not both.")
        return self


class DriverInterface(ModelInterface):
    def __init__(self, settings=None):
        super().__init__(settings)
        if "sdkpath" not in self.settings:
            raise RuntimeError(
                "Cannot instantiate tomato-picovna without supplying a sdkpath"
            )
        path = Path(self.settings["sdkpath"])
        if psutil.WINDOWS:
            path = path / "windows"
        elif psutil.LINUX:
            path = path / "linux_x64"
        else:
            raise RuntimeError("Unsupported OS")
        if sys.version_info[1] == 10:
            path = path / "python310"
        elif sys.version_info[1] == 11:
            path = path / "python311"
        sys.path.append(str(path))

        global vna
        vna = importlib.import_module("vna.vna")

    class DeviceManager(ModelInterface.DeviceManager):
        instrument: Any
        task_sweep_config: Any
        frequency_unit: str = "Hz"
        frequency_min: float
        frequency_max: float
        ports: set = {"S11"}

        _bandwidth: float = None
        _power_level: float = None
        _sweep_params: list[Sweep] = list()
        _sweep_nports: int = 1
        _last_sweep: dict = None

        @property
        def _temperature(self):
            return self.instrument.getTemperature()

        def __init__(
            self, driver: ModelInterface, key: tuple[str, int], **kwargs: dict
        ):
            super().__init__(driver, key, **kwargs)
            self.instrument = vna.Device.open(f"{key[1]}")
            info = self.instrument.getInfo()
            self.frequency_min = info.minSweepFrequencyHz
            self.frequency_max = info.maxSweepFrequencyHz
            self.task_sweep_config = None

        def attrs(self, **kwargs: dict) -> dict[str, Attr]:
            attrs_dict = {
                "temperature": Attr(type=float, units="Celsius", status=True),
                "bandwidth": Attr(type=float, units="Hz", rw=True),
                "power_level": Attr(type=float, units="dBm", rw=True),
                "sweep_params": Attr(type=Any, rw=True, status=True),
                "sweep_nports": Attr(type=int, rw=True, status=True),
                "last_sweep": Attr(type=dict, rw=False, status=False),
            }
            return attrs_dict

        def set_attr(self, attr: str, val: Any, **kwargs: dict):
            if attr not in self.attrs():
                raise ValueError(f"Unknown attr: {attr!r}")
            if not self.attrs()[attr].rw:
                raise ValueError(f"Read only attr: {attr!r}")
            if attr == "bandwidth":
                if val not in BANDWIDTH_SET:
                    raise ValueError(f"'bandwidth' of {val} is not permitted")
                setattr(self, f"_{attr}", val)
            elif attr == "power_level":
                setattr(self, f"_{attr}", val)
            elif attr == "sweep_nports":
                if val == 1:
                    self.ports = {"S11"}
                elif val == 2:
                    self.ports = {"S11", "S12", "S21", "S22"}
                else:
                    raise ValueError(f"'sweep_nports' has to be 1 or 2, not {val}")
                setattr(self, f"_{attr}", val)
            elif attr in {"sweep_params"}:
                self._sweep_params = [Sweep(**item) for item in val]

        def get_attr(self, attr: str, **kwargs: dict):
            if attr not in self.attrs():
                raise ValueError(f"Unknown attr: {attr!r}")
            if hasattr(self, f"_{attr}"):
                return getattr(self, f"_{attr}")

        def capabilities(self, **kwargs: dict) -> set:
            capabs = {"linear_sweep"}
            return capabs

        def prepare_task(self, task, **kwargs):
            super().prepare_task(task, **kwargs)
            self.task_sweep_config = self._build_sweep(
                self._sweep_params, self._power_level, self._bandwidth
            )

        def do_task(self, task: Task, **kwargs: dict):
            t0 = perf_counter()
            data = self._run_sweep_config(self.task_sweep_config)
            dt = perf_counter() - t0
            if dt > task.sampling_interval:
                logger.warning(
                    "'task.sampling_interval' of %f s is too short, last sweep took %f s",
                    task.sampling_interval,
                    dt,
                )
            else:
                logger.debug("last sweep took %f s", dt)
            # Merge Scans Here
            for k, v in data.items():
                self.data[k].append(v)

        @staticmethod
        def _build_sweep(
            sweep_params: list[Sweep], power_level: float, bandwidth: float
        ):
            logger.debug("building a sweep")
            mc = vna.MeasurementConfiguration()
            for sweep in sweep_params:
                if sweep.step is not None:
                    points = np.arange(sweep.start, sweep.stop + 1, sweep.step)
                elif sweep.points is not None:
                    points = np.linspace(sweep.start, sweep.stop, num=sweep.points)
                    points = np.around(points)
                logger.debug("adding a sweep section with %d points", len(points))
                for p in points:
                    pt = vna.MeasurementPoint()
                    pt.frequencyHz = p
                    pt.powerLeveldBm = power_level
                    pt.bandwidthHz = bandwidth
                    mc.addPoint(pt)
            logger.debug("sweep with %d total points built", len(mc.getPoints()))
            return mc

        def _run_sweep_config(self, sweep_config):
            data = {
                "uts": datetime.now().timestamp(),
                "temperature": self._temperature,
            }
            ret = self.instrument.performMeasurement(sweep_config)
            freq = []
            real = {k: [] for k in self.ports}
            imag = {k: [] for k in self.ports}
            for pt in ret:
                freq.append(pt.measurementFrequencyHz)
                for k in self.ports:
                    real[k].append(getattr(pt, k.lower()).real)
                    imag[k].append(getattr(pt, k.lower()).imag)
            data["freq"] = freq
            for k in self.ports:
                data[f"Re({k})"] = real[k]
                data[f"Im({k})"] = imag[k]
            self._last_sweep = data
            return data

    @in_devmap
    def task_data(self, key: tuple, **kwargs) -> Reply:
        data = self.devmap[key].get_data(**kwargs)

        if len(data) == 0:
            return Reply(success=False, msg="found no new datapoints")

        attrs = self.devmap[key].attrs(**kwargs)
        coords = {
            "uts": data.pop("uts"),
            "freq": (("uts", "freq"), data.pop("freq"), {"units": "Hz"}),
        }
        data_vars = {}
        for k, v in data.items():
            if k.startswith("Re") or k.startswith("Im"):
                data_vars[k] = (("uts", "freq"), v)
            else:
                units = {} if attrs[k].units is None else {"units": attrs[k].units}
                data_vars[k] = ("uts", v, units)
        ds = Dataset(data_vars=data_vars, coords=coords)
        return Reply(success=True, msg=f"found {len(data)} new datapoints", data=ds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(f"{vna=}")
    settings = {
        "sdkpath": r"C:\Users\Kraus\Downloads\picovna5_sdk_v_5_2_5\picovna5_sdk_v_5_2_5\python"
    }
    kwargs = dict(address="A0171", channel="11328")
    interface = DriverInterface(settings=settings)
    interface.dev_register(**kwargs)
    component = interface.devmap[("A0171", "11328")]
    print(f"{interface=}")
    print(f"{component=}")
    print(f"{vna=}")

    task = Task(
        component_tag="bla",
        max_duration=10,
        sampling_interval=1,
        technique_name="linear_sweep",
        technique_params={
            "bandwidth": 10_000,
            "power_level": -3,
            "sweep_params": [
                # dict(start=2_000_000_000, stop=2_500_000_000, points=101),
                dict(start=2_000_000_000, stop=2_500_000_000, step=5_000_000),
            ],
            "sweep_nports": 2,
        },
    )

    print(f"{interface.dev_get_attr(attr='last_sweep', **kwargs).data is None=}")
    interface.task_start(task=task, **kwargs)
    sleep(1)
    while interface.task_status(**kwargs).data["running"]:
        print(f"{interface.dev_get_attr(attr='last_sweep', **kwargs).data is None=}")
        sleep(1)
    data = interface.task_data(**kwargs)
    print(f"{data=}")
