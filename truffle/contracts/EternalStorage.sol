// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.7.0 <0.9.0;

import "./Owner.sol";

/** 
 * @title EternalStorage
 * @author Jun Lin (@jun-lsh)
 * @dev Storage of IPFS multihash based on project requirements, with ownership checking.
 * Code is adapted from https://github.com/saurfang/ipfs-multihash-on-solidity/blob/master/contracts/IPFSStorage.sol 
*/

contract EternalStorage is Owner {
    struct Multihash {
    bytes32 digest;
    uint8 hashFunction;
    uint8 size;
    }

    mapping(uint => mapping(bytes32 => Multihash)) MultihashStorage;

    function setHashValue(uint _dateVal, bytes32 _degdecVal, 
    bytes32 _digest, uint8 _hashFunction, uint8 _size)
    public 
    isOwner {
        Multihash memory entry = Multihash(_digest, _hashFunction, _size);
        MultihashStorage[_dateVal][_degdecVal] = entry;
    }

    function getHashValue(uint _dateVal, bytes32 _degdecVal)
    public
    view
    returns(bytes32 digest, uint8 hashfunction, uint8 size) 
    {
        Multihash memory entry = MultihashStorage[_dateVal][_degdecVal];
        return (entry.digest, entry.hashFunction, entry.size);
    }

    mapping(uint => uint) UIntStorageTest;

    function setUintValue(uint _index, uint _val)
    public
    isOwner {
        UIntStorageTest[_index] = _val;
    }

    function getUintValue(uint _index)
    public
    view
    returns(uint val)
    {
        return UIntStorageTest[_index];
    }

}