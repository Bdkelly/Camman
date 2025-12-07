import asyncio
from bleak import BleakScanner, BleakClient

class ESP32Controller:
    #Must Match ESP32 Code
    SERVICE_UUID = "SERVICE-CamMan-UUID"
    CHARACTERISTIC_UUID = "CHARACTERISTIC-CamMan-UUID"
    
    def __init__(self):
        self.client = None
        self.device = None

    async def scan_and_check(self):
        print("Scanning for ESP32...")
        device = await BleakScanner.find_device_by_filter(
            lambda d, ad: self.SERVICE_UUID.lower() in ad.service_uuids
        )
        if device:
             print(f"Found ESP32 at {device.address}")
             self.device = device
             return True
        return False
    
    async def connect(self):
        print("Scanning for ESP32...")
        if not self.device:
            self.device = await BleakScanner.find_device_by_filter(
            lambda d, ad: self.SERVICE_UUID.lower() in ad.service_uuids
        )

        if not self.device:
            print("Could not find ESP32 device with UUID.")
            return False
        print(f"Found ESP32 at {self.device.address}")
        self.client = BleakClient(self.device.address)
        
        try:
            await self.client.connect()
            print(f"Successfully connected to {self.device.address}")
            return self.client.is_connected
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.client = None
            return False