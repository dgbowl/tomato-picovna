import importlib
import logging
import sys
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal

import numpy as np
import pint
import xarray as xr
from pint import Quantity
from pydantic import BaseModel, model_validator
from pydantic_pint import PydanticPintQuantity
from tomato.driverinterface_2_1 import Attr, ModelDevice, ModelInterface
from tomato.driverinterface_2_1.decorators import coerce_val, log_errors, to_reply
from tomato.driverinterface_2_1.types import Val

pint.set_application_registry(pint.UnitRegistry(autoconvert_offset_to_baseunit=True))

ureg = pint.get_application_registry()

BANDWIDTH_SET = {
    10,
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    15_000,
    35_000,
    70_000,
    140_000,
}
POINTS_SET = {
    51,
    101,
    201,
    501,
    1_001,
    2_001,
    5_001,
    10_001,
}
logger = logging.getLogger(__name__)


def estimate_sweep_time(bw: int, npoints: int):
    # Parameters obtained using a curve fit to various npoints & bandwidths
    c0, c1, c2 = [7.17562593e-01, 1.83389054e00, 1.28624374e-04]
    return c0 + npoints * (c1 / bw + c2)


class Sweep(BaseModel):
    start: Annotated[Quantity, PydanticPintQuantity("Hz", strict=False, ureg=ureg)]
    stop: Annotated[Quantity, PydanticPintQuantity("Hz", strict=False, ureg=ureg)]
    points: Literal[*POINTS_SET] | None = None  # ty: ignore
    step: (
        Annotated[Quantity, PydanticPintQuantity("Hz", strict=False, ureg=ureg)] | None
    ) = None

    @model_validator(mode="after")
    def check_points_or_step(self):
        if self.points is None and self.step is None:
            raise ValueError("Must supply either 'points' or 'step'.")
        if self.points is not None and self.step is not None:
            raise ValueError("Must supply either 'points' or 'step', not both.")
        return self


class DriverInterface(ModelInterface):
    vna: ModuleType

    idle_measurement_interval = None

    def __init__(self, settings=None):
        super().__init__(settings)
        if "dllpath" not in self.settings:
            raise RuntimeError(
                "Cannot instantiate tomato-picovna without supplying a dllpath"
            )
        path = Path(self.settings["dllpath"])

        sys.path.append(str(path))
        logger.debug(f"{path=}")

        self.vna = importlib.import_module("vna")

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
    ports: set

    bandwidth: pint.Quantity
    power_level: pint.Quantity
    sweep_params: Sweep
    sweep_nports: int
    calibration: str | None

    @property
    def temperature(self) -> Quantity:
        temp = self.instrument.getTemperature()
        return Quantity(temp, "celsius")

    def __init__(self, driver: DriverInterface, key: tuple[str, str], **kwargs: dict):
        assert driver.vna is not None
        # Will raise vna.vna.DeviceNotFoundException if channel is incorrect
        _address, channel = key
        self.instrument = driver.vna.Device.open(channel)
        info = self.instrument.getInfo()
        self.frequency_min = Quantity(info.minSweepFrequencyHz, "Hz")
        self.frequency_max = Quantity(info.maxSweepFrequencyHz, "Hz")
        self.task_sweep_config = None
        self.bandwidth = Quantity(max(BANDWIDTH_SET), "Hz")
        self.power_level = Quantity("-3 dBm")
        self.sweep_params = Sweep(
            start=self.frequency_min,
            stop=self.frequency_max,
            points=min(POINTS_SET),
        )
        self.sweep_nports = 1
        self.ports = {"S11"}
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
            "sweep_params": Attr(type=Sweep, rw=True, status=True),
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
            self.sweep_params = val
            self.task_sweep_config = self._build_sweep(
                self.sweep_params,
                self.power_level.to("dBm").m,
                self.bandwidth.to("Hz").m,
                self.driver.vna,  # ty: ignore
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
            self.sweep_params,
            self.power_level.to("dBm").m,
            self.bandwidth.to("Hz").m,
            self.driver.vna,  # ty: ignore
        )

    def do_measure(self, **kwargs: dict):
        logger.debug("performing measurement")
        coords = {"uts": (["uts"], [datetime.now(UTC).timestamp()])}
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
    def _build_sweep(
        sweep: Sweep, power_level: float, bandwidth: float, vna: ModuleType
    ):
        logger.debug("building a sweep")
        mc = vna.MeasurementConfiguration()
        if sweep.step is not None:
            points = np.arange(
                sweep.start.to("Hz").m,
                sweep.stop.to("Hz").m + 1,
                sweep.step.to("Hz").m,
            )
        elif sweep.points is not None:
            points = np.linspace(
                sweep.start.to("Hz").m,
                sweep.stop.to("Hz").m,
                num=sweep.points,
            )
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
