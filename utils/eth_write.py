import web3
from utils.utils import *
import json

w3 = None
eternalStorage = None
eth_keys = []

contract_address = "0x3260Df12C458Ac84CBbeFb82F92E8Ddc57927CD7"
contract_json = "./../truffle/build/contracts/EternalStorage.json"

def initKeys():
    global eth_keys
    eth_keys = get_keys("./../keys/api_keys.json", ["OWNER_ADDRESS", "OWNER_PRIVATE_KEY", "INFURA_API_KEY"])


def initWeb3(provider, contract_address, contract_json):
    global w3
    global eternalStorage
    w3 = web3.Web3(web3.HTTPProvider(provider))
    eternalStorage = w3.eth.contract(address = contract_address, abi = json.load(open(contract_json))["abi"])

def appendUint(key_num, val_num):
    global eth_keys
    nonce = w3.eth.getTransactionCount(eth_keys[0])
    tx_dict = eternalStorage.functions.setUintValueB32(key_num, val_num).build_transaction({
        "from":eth_keys[0],
        "nonce":nonce
    })
    signed_tx = w3.eth.account.sign_transaction(tx_dict, eth_keys[1])
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

def appendHash(timestamp, coords, ipfs_hash):
    global eth_keys
    nonce = w3.eth.getTransactionCount(eth_keys[0])
    ipfs_processed = get_bytes32_from_ipfs(ipfs_hash)
    print(f'[+] Processed IPFS Hash {ipfs_processed}')
    print(f'[+] Coords {hex(int.from_bytes(get_bytes32_from_coords(coords),"little"))}')
    tx_dict = eternalStorage.functions.setHashValue(timestamp, 
        get_bytes32_from_coords(coords), 
        ipfs_processed[0], ipfs_processed[1], ipfs_processed[2]
    ).build_transaction({
        "from":eth_keys[0],
        "nonce":nonce
    })
    signed_tx = w3.eth.account.sign_transaction(tx_dict, eth_keys[1])
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)
    print(tx_hash)

if __name__ == "__main__":
    initKeys()
    initWeb3(f"https://ropsten.infura.io/v3/{eth_keys[2]}", contract_address, contract_json)
#    appendUint(bytes.fromhex("abab"), 89)
    appendHash(6, [[0,1]]*4, "QmfM9xSr5g344Uhxt2kzxb4EoBqy4hy6zGPJEYD4MUCucu")
    


