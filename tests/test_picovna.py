import logging
import time

from tomato.driverinterface_2_1 import Task

from tomato_picovna import DriverInterface

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    settings = {
        "dllpath": r"/opt/picovna/lib/",
        "calibration": "/home/kraus/Documents/Instruments/COCoS/calibrations/2026-08-27_5500MHz-7500MHz_10kHz_3dBm_03.calx",
    }
    kwargs = dict(address="A0165", channel="10708")
    interface = DriverInterface(settings=settings)
    print(f"{interface=}")
    print(f"{interface.cmp_register(**kwargs)=}")
    component = interface.devmap[("A0165", "10708")]
    print(f"{component=}")
    print(f"{component.calibration=}")

    sweep_params = dict(start=5_500_000_000, stop=7_500_000_000, points=10000)

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

    print(f"{interface.task_start(task=task, **kwargs)=}")
    time.sleep(5)
    while True:
        ret = interface.cmp_status(**kwargs)
        print(f"{ret=}")
        if ret.data["running"] is False:
            break
        time.sleep(1)
    ret = interface.task_data(**kwargs)
    print(f"{ret=}")
    ret.data.to_netcdf("4.nc", engine="h5netcdf")

if False:
    task = Task(
        component_role="bla",
        max_duration=10,
        sampling_interval=5,
        technique_name="linear_sweep",
        task_params={
            "bandwidth": 10_000,
            "power_level": -3,
            "sweep_params": [
                dict(start=2_500_000_000, stop=3_000_000_000, points=2500),
                dict(start=4_500_000_000, stop=5_000_000_000, points=2500),
                dict(start=5_800_000_000, stop=6_800_000_000, points=5001),
            ],
            "sweep_nports": 1,
        },
    )
    print(f"{interface.task_start(task=task, **kwargs)=}")
    while True:
        time.sleep(0.1)
        ret = interface.cmp_status(**kwargs)
        print(f"{ret=}")
        if ret.data["running"] is False:
            break
    ret = interface.task_data(**kwargs)
    ret.data.to_netcdf("split_calx.nc", engine="h5netcdf")
