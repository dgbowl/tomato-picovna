
from tomato_picovna import DriverInterface, vna, Task
import time
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(f"{vna=}")
    settings = {
        "sdkpath": r"C:\Users\Kraus\Documents\Instruments\COCoS\picovna5_sdk_v_5_2_5\python",
        "calibration": r"C:\Users\Kraus\Documents\Instruments\COCoS\2025-05-22-Calibration_Tomato\2025-05-22_2.5-7.5GHz_10kHz_-3dBm_picovna5.calx",
    }
    kwargs = dict(address="A0165", channel="10708")
    interface = DriverInterface(settings=settings)
    interface.dev_register(**kwargs)
    component = interface.devmap[("A0165", "10708")]
    print(f"{interface=}")
    print(f"{component=}")
    print(f"{vna=}")
    print(f"{component.calibration=}")

    sweep_params = [
        # dict(start=2_000_000_000, stop=2_500_000_000, points=101),
        dict(start=2_700_000_000, stop=6_700_000_000, points=10001),
    ]

    task = Task(
        component_role="bla",
        max_duration=10,
        sampling_interval=5,
        technique_name="linear_sweep",
        task_params={
            "bandwidth": 10_000,
            "power_level": -3,
            "sweep_params": sweep_params,
            "sweep_nports": 1,
        },
    )

    print(f"{interface.cmp_set_attr(**kwargs, attr='sweep_params', val=sweep_params)=}")
    print(f"{interface.cmp_measure(**kwargs)=}")
    # print(f"{interface.dev_get_attr(attr='last_sweep', **kwargs).data is None=}")
    # interface.task_start(task=task, **kwargs)
    while True:
        time.sleep(0.1)
        ret = interface.cmp_status(**kwargs)
        print(f"{ret=}")
        if ret.data["running"] is False:
            break
    ret = interface.cmp_last_data(**kwargs)
    print(f"{ret=}")
    ret.data.to_netcdf("cmp_measure_calx_inter_3.nc", engine="h5netcdf")

    assert False

    print(f"{interface.task_start(task=task, **kwargs)=}")
    while True:
        time.sleep(0.1)
        ret = interface.cmp_status(**kwargs)
        print(f"{ret=}")
        if ret.data["running"] is False:
            break
    ret = interface.task_data(**kwargs)
    ret.data.to_netcdf("task_calx.nc", engine="h5netcdf")

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