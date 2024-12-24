import pyudev
import glob


def get_device_info(dev_path):
    context = pyudev.Context()
    device = pyudev.Devices.from_device_file(context, dev_path)
    info = {
        'DEVNAME': device.device_node,
        'ID_MODEL': device.get('ID_MODEL', 'Unknown'),
        'ID_VENDOR': device.get('ID_VENDOR', 'Unknown'),
        'ID_SERIAL': device.get('ID_SERIAL', 'Unknown')
    }
    return info


def find_device_by_serial(target_serial):
    devices = glob.glob('/dev/ttyUSB*')
    for dev in devices:
        info = get_device_info(dev)
        if info['ID_SERIAL'] == target_serial:
            return info['DEVNAME']
    return None

