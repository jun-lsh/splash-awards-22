<template>
  <v-container fluid class="px-12 py-6 fill-height">
    <v-row class="d-flex align-center justify-center fill-height">
      <v-col cols="6" style="height: 100%">
        <iframe
          id="leaflet"
          src="leaflet.html"
          frameborder="0"
          width="100%"
          height="75%"
        />
      </v-col>
    </v-row>
  </v-container>
</template>

<script lang="ts">
import Vue from "vue";
import Web3 from "web3";
import ipfs_utils from "@/plugins/ipfs_decode";
import { Multihash } from "@/types/multihash";

export default Vue.extend({
  name: "Home",
  data: () =>(
    {
      CONTRACT_ADDRESS: "0x3260Df12C458Ac84CBbeFb82F92E8Ddc57927CD7",
      iframe_created: false,
    }
  ),
  methods: {
    receiveMessage(event) {
      if (event.data === "idle") {
        console.log("POC communication between iframe and parent");
      }
      if (event.data === "created") {
        console.log("iframe created");
        this.iframe_created = true;
      }
    },

    callIframe() {
      if (this.iframe_created) {
        const iframe = document.querySelector("iframe");
        if (iframe != null && iframe.contentWindow != null) {
          iframe.contentWindow.postMessage("data", "*");
        }
      }
      else {
        console.log("iframe not created, cant send message");
      }
    }
  },
  mounted () {
    window.addEventListener("message", this.receiveMessage);
  },
  beforeDestroy () {
    window.removeEventListener("message", this.receiveMessage);
  },
  created(){
    const web3 = new Web3(
      new Web3.providers.HttpProvider(
        `https://ropsten.infura.io/v3/${process.env.VUE_APP_INFURA_API_KEY}`
    )
    );
    const eternalStorageJson = require("./../../../truffle/build/contracts/EternalStorage.json");
    const eternalStorage = new web3.eth.Contract(eternalStorageJson.abi, this.CONTRACT_ADDRESS);
    console.log("called");
    const value = "0x0";
    eternalStorage.methods.getHashValue(1, value).call().then(
      function(multihash : Multihash){
        let ipfs_hash = ipfs_utils.decode(multihash);
        console.log(ipfs_hash);
        fetch("https://gateway.pinata.cloud/ipfs/" + ipfs_hash, {
          headers: {
            "accept": "application/json",
          }
        }).then(
          response => response.json().then(
            data => {
              console.log(data);
            }
          )
        );
      }
    );
  }
});
</script>
