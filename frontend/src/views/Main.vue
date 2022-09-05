<template>
  <v-container fluid>
    <v-card>
      <v-card-title>
        <div v-if="name === ''">
          Your profile
        </div>
        <div v-else>
          Welcome to Vue, {{ name }}
        </div>
        <v-spacer/>
      </v-card-title>
      <div class="pa-8">
        <v-text-field
          v-model="name"
          label="Name"
        >
        </v-text-field>
      </div>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import Vue from "vue";
import Web3 from 'web3';
import ipfs_utils from '@/plugins/ipfs_decode'
import { Multihash } from '@/types/multihash'

export default Vue.extend({
  name: "Main",
  data: () =>(
    {
      name: "",
      mainOut: "",
      CONTRACT_ADDRESS: "0x3260Df12C458Ac84CBbeFb82F92E8Ddc57927CD7"
    }
  ),
  created(){

    const web3 = new Web3(  
      new Web3.providers.HttpProvider(
        `https://ropsten.infura.io/v3/${process.env.VUE_APP_INFURA_API_KEY}`
    )
    );
    const eternalStorageJson = require("./../../../truffle/build/contracts/EternalStorage.json");
    const eternalStorage = new web3.eth.Contract(eternalStorageJson.abi, this.CONTRACT_ADDRESS)
    
    const value = "0x0";
    eternalStorage.methods.getHashValue(0, value).call().then(
      function(data : Multihash){
        console.log(ipfs_utils.decode(data));
      }
    )
    
  }
});
</script>
