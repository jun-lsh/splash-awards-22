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
import Web3 from "web3";

export default Vue.extend({
  name: "Main",
  data: () =>(
    {
      name: "",
      mainOut: "",
      CONTRACT_ADDRESS: "0x0115f0ce4a46264919cCba40aE6f2911F2Cfb786"
    }
  ),
  created(){
    const web3 = new Web3("ws://127.0.0.1:8545");
    const eternalStorageJson = require("../truffle/build/contracts/EternalStorage.json");
    const eternalStorage = new web3.eth.Contract(eternalStorageJson.abi, this.CONTRACT_ADDRESS);
    console.log("Called");
    web3.eth.getAccounts().then(
      data => {console.log(data);}
    );

    const value = 100;
    eternalStorage.methods.getUintValue(value).call().then(
      function(data : Object){
        console.log(data);
      }
    );
  }
});
</script>
