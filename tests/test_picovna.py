import logging
import time

from tomato.driverinterface_2_1 import Task
from tomato.models import Component

from tomato_picovna import DriverInterface

ADDR = None
CHAN = "10708"
# CHAN = "DEMO000"

cmp = Component(driver="picovna", address=ADDR, channel=CHAN, device="vna")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(f"{cmp=}")
    settings = {
        "dllpath": r"/opt/picovna/lib/",
        # "calibration": "/home/kraus/Documents/Instruments/COCoS/calibrations/2026-08-27_5500MHz-7500MHz_10kHz_3dBm_03.calx",
    }
    interface = DriverInterface(settings=settings)
    print(f"{interface=}")
    print(f"{interface.cmp_register(**cmp.model_dump(exclude={'driver'}))=}")
    component = interface.devmap[cmp.name]
    print(f"{component=}")
    print(f"{component.driver.settings=}")

    sweep_params = {"start": 5_500_000_000, "stop": 7_500_000_000, "points": 1001}

    task = Task(
        component_role="bla",
        max_duration=4,
        sampling_interval=2,
        technique_name="linear_sweep",
        task_params={
            "bandwidth": 10_000,
            "power_level": -3,
            "sweep_params": sweep_params,
            "sweep_nports": 1,
        },
    )
    print(f"{task=}")

    print(f"{interface.task_start(task=task, name=cmp.name)=}")
    time.sleep(5)
    while True:
        ret = interface.cmp_status(name=cmp.name)
        print(f"{ret=}")
        if ret.data.state != "task":
            break
        time.sleep(1)
    ret = interface.task_data(name=cmp.name)
    print(f"{ret=}")
    ret.data.to_netcdf("4.nc", engine="h5netcdf")
