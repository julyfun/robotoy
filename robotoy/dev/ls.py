import glob
from robotoy.dev import get_device_info

def main():
    devices = glob.glob('/dev/ttyUSB*')
    for dev in devices:
        info = get_device_info(dev)
        print(f"Device: {info['DEVNAME']}")
        print(f"  Model: {info['ID_MODEL']}")
        print(f"  Vendor: {info['ID_VENDOR']}")
        print(f"  Serial: {info['ID_SERIAL']}")
        print()

if __name__ == '__main__':
    main()

