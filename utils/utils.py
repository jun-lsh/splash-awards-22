import json
import base58

def get_keys(path, keys):
    output_keys = []
    with open(path) as f:
        json_file = json.load(f)
        for k in keys:
            output_keys.append(json_file[k])
    return output_keys

def get_bytes32_from_ipfs(ipfs_multihash):
    # ipfs hashes can be b58 decoded, the first element being the hashFunc, second being the size
    # the remainder of the bytes is the hash digest
    b58_decoded = base58.b58decode(ipfs_multihash)
    return (b58_decoded[2:], b58_decoded[0], b58_decoded[1])

def get_bytes32_from_coords(coords):
    # following decimal degrees specification
    # 4 coordinates being passed in would be in the form of [+-90, +-180] --> 18003600
    output_bytes = 0
    for coord in coords:
        output_bytes <<= 64
        output_bytes += (coord[0] << 16) + coord[1]
    return output_bytes.to_bytes(32, 'big')
