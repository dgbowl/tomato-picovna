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
    def check_points_or_every(self):
        if self.points is None and self.step is None:
            raise ValueError("Must supply either 'points' or 'every'.")
        if self.points is not None and self.step is not None:
            raise ValueError("Must supply either 'points' or 'every', not both.")
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
        measurement: Any
        frequency_unit: str = "Hz"
        frequency_min: float
        frequency_max: float

        _bandwidth: float = None
        _power_level: float = None
        _sweep_params: list[Sweep] = list()

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

        def attrs(self, **kwargs: dict) -> dict[str, Attr]:
            attrs_dict = {
                "temperature": Attr(type=float, units="Celsius", status=True),
                "bandwidth": Attr(type=float, units="Hz", rw=True),
                "power_level": Attr(type=float, units="dBm", rw=True),
                "sweep_params": Attr(type=Any, rw=True, status=True),
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
            self._build_sweep()

        def do_task(self, task: Task, **kwargs: dict):
            uts = datetime.now().timestamp()
            t0 = perf_counter()
            ret = self.instrument.performMeasurement(self.measurement)
            dt = perf_counter() - t0
            if dt > task.sampling_interval:
                logger.warning(
                    "'task.sampling_interval' of %f s is too short, last sweep took %f s",
                    task.sampling_interval,
                    dt,
                )
            else:
                logger.debug("last sweep took %f s", dt)
            self.data["uts"].append(uts)
            self.data["temperature"].append(self._temperature)
            freq = []
            real = []
            imag = []
            for pt in ret:
                freq.append(pt.measurementFrequencyHz)
                real.append(pt.s11.real)
                imag.append(pt.s11.imag)
            self.data["freq"].append(freq)
            self.data["Re(S11)"].append(real)
            self.data["Im(S11)"].append(imag)

        def _build_sweep(self):
            logger.debug("building a sweep")
            mc = vna.MeasurementConfiguration()
            for sweep in self._sweep_params:
                if sweep.step is not None:
                    points = np.arange(sweep.start, sweep.stop + 1, sweep.step)
                elif sweep.points is not None:
                    points = np.linspace(sweep.start, sweep.stop, num=sweep.points)
                    points = np.around(points)
                logger.debug("adding a sweep section with %d points", len(points))
                for p in points:
                    pt = vna.MeasurementPoint()
                    pt.frequencyHz = p
                    pt.powerLeveldBm = self._power_level
                    pt.bandwidthHz = self._bandwidth
                    mc.addPoint(pt)
            logger.debug("sweep with %d total points built", len(mc.getPoints()))
            self.measurement = mc

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
        sampling_interval=5,
        technique_name="linear_sweep",
        technique_params={
            "bandwidth": 100,
            "power_level": -3,
            "sweep_params": [
                # dict(start=2_000_000_000, stop=2_500_000_000, points=101),
                dict(start=2_000_000_000, stop=2_500_000_000, step=5_000_000),
            ],
        },
    )

    interface.task_start(task=task, **kwargs)
    sleep(1)
    while interface.task_status(**kwargs).data["running"]:
        print(f"{interface.task_status(**kwargs)=}")
        sleep(1)
    data = interface.task_data(**kwargs)
    print(f"{data=}")
