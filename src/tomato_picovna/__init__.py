from typing import Any, Optional
from types import ModuleType
from tomato.driverinterface_2_1 import ModelInterface, ModelDevice, Attr
from tomato.driverinterface_2_1.decorators import coerce_val, log_errors, to_reply
from tomato.driverinterface_2_1.types import Val
from pathlib import Path
import psutil
import sys
import importlib
from pydantic import BaseModel, model_validator
import numpy as np
import logging
from datetime import datetime
import xarray as xr
import pint
import time

pint.set_application_registry(pint.UnitRegistry(autoconvert_offset_to_baseunit=True))
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
    idle_measurement_interval = None

    def __init__(self, settings=None):
        super().__init__(settings)
        if "dllpath" not in self.settings:
            raise RuntimeError(
                "Cannot instantiate tomato-picovna without supplying a dllpath"
            )
        path = Path(self.settings["dllpath"])
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

    def DeviceFactory(self, key, **kwargs):
        return Device(self, key, **kwargs)

    @log_errors
    @to_reply
    def cmp_register(
        self, address: str, channel: str, **kwargs: dict
    ) -> tuple[bool, str, set]:
        key = (address, channel)
        self.devmap[key] = self.DeviceFactory(key, **kwargs)
        capabs = self.devmap[key].capabilities()
        self.retries[key] = 0
        return (True, f"device {key!r} registered", capabs)


class Device(ModelDevice):
    instrument: Any
    task_sweep_config: Any
    frequency_unit: str = "Hz"
    frequency_min: pint.Quantity
    frequency_max: pint.Quantity
    ports: set = {"S11"}

    bandwidth: pint.Quantity
    power_level: pint.Quantity
    sweep_params: list[Sweep]
    sweep_nports: int
    calibration: str

    @property
    def temperature(self) -> pint.Quantity:
        return pint.Quantity(self.instrument.getTemperature(), "celsius")

    def __init__(self, driver: ModelInterface, key: tuple[str, str], **kwargs: dict):
        # Will raise vna.vna.DeviceNotFoundException if channel is incorrect
        address, channel = key
        self.instrument = vna.Device.open(channel)
        info = self.instrument.getInfo()
        self.frequency_min = pint.Quantity(info.minSweepFrequencyHz, "Hz")
        self.frequency_max = pint.Quantity(info.maxSweepFrequencyHz, "Hz")
        self.task_sweep_config = None
        self.bandwidth = pint.Quantity("140 kHz")
        self.power_level = pint.Quantity("-3 dBm")
        self.sweep_params = list()
        self.sweep_nports = 1
        if "calibration" in driver.settings:
            self.calibration = driver.settings["calibration"]
        else:
            self.calibration = None
        super().__init__(driver, key, **kwargs)

    def attrs(self, **kwargs: dict) -> dict[str, Attr]:
        attrs_dict = {
            "temperature": Attr(type=pint.Quantity, units="celsius", status=False),
            "bandwidth": Attr(type=pint.Quantity, units="Hz", rw=True),
            "power_level": Attr(type=pint.Quantity, units="dBm", rw=True),
            "sweep_params": Attr(type=list, rw=True, status=True),
            "sweep_nports": Attr(type=int, rw=True, status=True),
        }
        return attrs_dict

    @coerce_val
    def set_attr(self, attr: str, val: Any, **kwargs: dict) -> Val:
        if attr == "bandwidth":
            if val.to("Hz").m not in BANDWIDTH_SET:
                raise ValueError(f"'bandwidth' of {val} is not permitted")
            self.bandwidth = val
        elif attr == "sweep_nports":
            if val == 1:
                self.ports = {"S11"}
            elif val == 2:
                self.ports = {"S11", "S12", "S21", "S22"}
            else:
                raise ValueError(f"'sweep_nports' has to be 1 or 2, not {val}")
            self.sweep_nports = val
        elif attr == "power_level":
            self.power_level = val
        elif attr == "sweep_params":
            self.sweep_params = [Sweep(**item) for item in val]
            self.task_sweep_config = self._build_sweep(
                self.sweep_params,
                self.power_level.to("dBm").m,
                self.bandwidth.to("Hz").m,
            )
        return val

    def get_attr(self, attr: str, **kwargs: dict) -> Val:
        if attr not in self.attrs():
            raise AttributeError(f"Unknown attr: {attr!r}")
        return getattr(self, attr)

    def capabilities(self, **kwargs: dict) -> set:
        capabs = {"linear_sweep"}
        return capabs

    def prepare_task(self, task, **kwargs):
        super().prepare_task(task, **kwargs)
        logger.critical("loading calibration")
        if self.calibration is not None:
            self.instrument.applyCalibrationFromFile(self.calibration)
        else:
            self.instrument.loadFactoryCalibration()
        logger.critical("building sweep")
        self.task_sweep_config = self._build_sweep(
            self.sweep_params, self.power_level.to("dBm").m, self.bandwidth.to("Hz").m
        )

    def do_measure(self, **kwargs: dict):
        logger.debug("performing measurement")
        coords = {"uts": (["uts"], [datetime.now().timestamp()])}
        temperature = self.temperature
        data_vars = {
            "temperature": (["uts"], [temperature.m], {"units": str(temperature.u)}),
        }

        # ret = self.instrument.performMeasurement(self.task_sweep_config)
        am = self.instrument.startMeasurement(self.task_sweep_config)
        bw = self.bandwidth.to("Hz").m
        npoints = self.task_sweep_config.numPoints()
        time.sleep(estimate_sweep_time(bw, npoints))
        ret = am.getAllPoints()

        freq = []
        real = {k: [] for k in self.ports}
        imag = {k: [] for k in self.ports}
        for pt in ret:
            freq.append(pt.measurementFrequencyHz)
            for k in self.ports:
                real[k].append(getattr(pt, k.lower()).real)
                imag[k].append(getattr(pt, k.lower()).imag)
        coords["freq"] = (["freq"], freq, {"units": self.frequency_unit})
        for k in self.ports:
            data_vars[f"Re({k})"] = (["uts", "freq"], [real[k]])
            data_vars[f"Im({k})"] = (["uts", "freq"], [imag[k]])
        self.last_data = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )
        logger.debug("measurement done")

    @staticmethod
    def _build_sweep(sweep_params: list[Sweep], power_level: float, bandwidth: float):
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
