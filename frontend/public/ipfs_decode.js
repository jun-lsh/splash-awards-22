import bs58 from 'bs58'
  
var ipfs_utils = {
    decode(multihash){
        let hashBytes = Buffer.from(multihash.digest.slice(2), 'hex')
        let multihashBytes = new(hashBytes.constructor)(2+hashBytes.length);
        multihashBytes[0] = multihash.hashfunction;
        multihashBytes[1] = multihash.size;
        multihashBytes.set(hashBytes, 2);

        let outputHash = bs58.encode(multihashBytes);
        return outputHash
    }
}

export default ipfs_utils