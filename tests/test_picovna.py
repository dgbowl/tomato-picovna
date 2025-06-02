from tomato_picovna import DriverInterface, vna
from tomato.driverinterface_2_1 import Task
import time
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(f"{vna=}")
    settings = {
        "sdkpath": r"C:\Users\kraus\AppData\Local\dgbowl\tomato\picovna5_sdk_v_5_2_5\python",
        "calibration": r"C:\Users\kraus\AppData\Local\dgbowl\tomato\picovna5_sdk_v_5_2_5\2025-05-22_2.5-7.5GHz_10kHz_-3dBm_picovna5.calx",
    }
    kwargs = dict(address="A0165", channel="10708")
    interface = DriverInterface(settings=settings)
    print(f"{interface=}")
    print(f"{interface.cmp_register(**kwargs)=}")
    component = interface.devmap[("A0165", "10708")]
    print(f"{component=}")
    print(f"{vna=}")
    print(f"{component.calibration=}")

if False:
    sweep_params = [
        # dict(start=2_000_000_000, stop=2_500_000_000, points=101),
        dict(start=2_700_000_000, stop=6_700_000_000, points=10001),
    ]

    task = Task(
        component_role="bla",
        max_duration=300,
        sampling_interval=10,
        technique_name="linear_sweep",
        task_params={
            "bandwidth": 10_000,
            "power_level": -3,
            "sweep_params": sweep_params,
            "sweep_nports": 1,
        },
    )

    print(f"{interface.task_start(task=task, **kwargs)=}")
    while True:
        ret = interface.cmp_status(**kwargs)
        print(f"{ret=}")
        if ret.data["running"] is False:
            break
        time.sleep(30)
    ret = interface.task_data(**kwargs)
    ret.data.to_netcdf("QLCZES_2.nc", engine="h5netcdf")

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
