import usb
from sys import platform as _platform
import time

dev = usb.core.find(idVendor=0x1209, idProduct=0xada1)
if not dev:
    print("Could not find device")
    exit(1)

#print(dev)

#if _platform == "linux" or _platform == "linux2":
reattach = False
interface = 0
if dev.is_kernel_driver_active(interface):
    reattach = True
    #dev.detach_kernel_driver(interface)
    print("kernel driver active")

#dev.reset()
#cfg = dev.get_active_configuration()
#print(cfg)
#time.sleep(1)
#dev.set_configuration()
#time.sleep(1)
#dev.ctrl_transfer(0x40 , 0x1, 0, 0, [])
#dev.ctrl_transfer(0b01000000 , 0x1, 0, 0, [])
try:
    dev.ctrl_transfer(bmRequestType=0b11000000, bRequest=0x0)
    pass
except Exception as e:
    print(f"Could not send ILA request: {e}")

time.sleep(1)
#dev.ctrl_transfer(0b01000000, 0x1, 0, 0, [])
try:
    ret = dev.ctrl_transfer(bmRequestType=0b11000000, bRequest=0x1) #VENDOR (2) request; TOGGLE_CONVOLUTION (1)
    #ret = dev.ctrl_transfer(bmRequestType=0b10100001, bRequest=0x02, wIndex=2, data_or_wLength=4) #CLASS (1) request RANGE(2)
    #ret = dev.ctrl_transfer(bmRequestType=0b10100000, bRequest=0x01, data_or_wLength=4)  # CLASS (1) request CUR(1)
    print(f"Result: {ret}")
except Exception as e:
    print(f"Could not send TOGGLE_CONVOLUTION request: {e}")


#dev.ctrl_transfer(0x40 , 0x03, 1, 0, [])
if reattach:
    print("reattaching")
    #dev.attach_kernel_driver(interface)
