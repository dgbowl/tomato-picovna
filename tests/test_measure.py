import logging
import time

from tomato.models import Component

from tomato_picovna import DriverInterface

ADDR = None
CHAN = "10708"
# CHAN = "DEMO000"

cmp = Component(driver="picovna", address=ADDR, channel=CHAN, device="vna")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(f"{cmp=}")
    settings = {"dllpath": r"/opt/picovna/lib/"}
    interface = DriverInterface(settings=settings)
    print(f"{interface=}")
    print(f"{interface.cmp_register(**cmp.model_dump(exclude={'driver'}))=}")
    print(f"{interface.cmp_measure(name=cmp.name)=}")
    while True:
        time.sleep(0.2)
        ret = interface.cmp_status(name=cmp.name)
        print(f"{ret=}")
        if ret.data.state == "idle":
            break
    print(f"{interface.cmp_last_data(name=cmp.name)=}")
