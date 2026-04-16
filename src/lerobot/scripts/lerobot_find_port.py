# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to find the USB port associated with your MotorsBus.

Example:

```shell
lerobot-find-port
```
"""

import platform
import time
from pathlib import Path


def find_available_ports():
    from lerobot.utils.import_utils import require_package

    require_package("pyserial", extra="hardware", import_name="serial")
    from serial.tools import list_ports

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def get_stable_id_map() -> dict[str, str]:
    """Build a map from resolved device path -> /dev/serial/by-id/ symlink path."""
    by_id = Path("/dev/serial/by-id")
    if not by_id.exists():
        return {}
    result = {}
    for link in by_id.iterdir():
        result[str(link.resolve())] = str(link)
    return result


def find_port():
    print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    stable_ids_before = get_stable_id_map()
    print("Ports before disconnecting:", ports_before)

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    input()  # Wait for user to disconnect the device

    time.sleep(0.5)  # Allow some time for port to be released
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        print(f"The port of this MotorsBus is '{port}'")
        stable_id = stable_ids_before.get(str(Path(port).resolve()))
        if stable_id:
            print(f"Persistent device path (recommended): '{stable_id}'")
        print("Reconnect the USB cable.")
    elif len(ports_diff) == 0:
        raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


def main():
    find_port()


if __name__ == "__main__":
    main()
