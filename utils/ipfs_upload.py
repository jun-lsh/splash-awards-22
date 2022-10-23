from utils.utils import get_keys
from utils.pinatapy import PinataPy

pinata = None


def initPinata():
    global pinata
    pinata_keys = get_keys("./../keys/api_keys.json", ["PINATA_API_KEY", "PINATA_API_SECRET"])
    pinata = PinataPy(pinata_keys[0], pinata_keys[1])


def pinFile(path):
    result = pinata.pin_file_to_ipfs(path)
    hash = result["IpfsHash"]
    return hash

# remember to run this script from the root directory
if __name__ == "__main__":

    initPinata()
    hash = pinFile("base_heatmap.json")

    file = open("hashes.txt", "a")
    file.write(f"{hash}\n")
    file.close()
