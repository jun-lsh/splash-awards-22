from utils import get_keys
from pinatapy import PinataPy

pinata = None

def initPinata():
    global pinata
    pinata_keys = get_keys("./../api_keys.json", ["PINATA_API_KEY", "PINATA_API_SECRET"])
    pinata = PinataPy(pinata_keys[0], pinata_keys[1])

def pinFile(path):
    result = pinata.pin_file_to_ipfs(path)
    hash = result["IpfsHash"]
    return hash

if __name__ == "__main__":
    initPinata()
    pinFile("base_heatmap.json")
