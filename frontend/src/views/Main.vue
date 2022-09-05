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

export default Vue.extend({
  name: "Main",
  data: () =>(
    {
      name: "",
      mainOut: "",
      CONTRACT_ADDRESS: "0x79844ed92837B1E8a0BAfE33F582adE294D3F078"
    }
  ),
  created(){
    
    console.log(`https://ropsten.infura.io/v3/${process.env.VUE_APP_INFURA_API_KEY}`);
    const web3 = new Web3(  
      new Web3.providers.HttpProvider(
        `https://ropsten.infura.io/v3/${process.env.VUE_APP_INFURA_API_KEY}`
    )
    );
    const eternalStorageJson = require("./../../../truffle/build/contracts/EternalStorage.json");
    const eternalStorage = new web3.eth.Contract(eternalStorageJson.abi, this.CONTRACT_ADDRESS)
    console.log("Called");
    web3.eth.getAccounts().then(
      data => {console.log(data);}
    )

    const value = "0xabab";
    eternalStorage.methods.getUintValueB32(value).call().then(
      function(data : Object){
        console.log(data);
      }
    )
    
  }
});
</script>
