<template>
  <v-container fluid class="px-12 py-6 fill-height">
    <v-row class="d-flex align-center justify-center fill-height">
      <v-col cols="4">
          <h2>
            What Does This Map Show Me?
          </h2>
        <p style="text-align: justify; text-justify: inter-word;">
          The map displays the <b>concentration of microplastics in the ocean</b>
          and using <b>deep learning</b>, is able to <b>predict the motion and concentration of
          microplastics in the future</b>. In shorter timescales, this map would be helpful to
          facilitate cleanup efforts in the ocean by <b>highlighting areas of high microplastics concentration</b>,
          which is directly correlated with general plastic pollution in the region.
          In longer timescales, we would be able to see <b>the dangers if this problem is left unchecked</b>,
          allowing us to see how our oceans may end up if we do not cut down on plastic waste.


          This map is predicting the microplastics concentration for <b>15 September, 2022</b>
        </p>
      </v-col>
      <v-col cols="8" style="height: 70%">
        <iframe
          id="leaflet"
          src="leaflet.html"
          frameborder="0"
          width="100%"
          height="100%"
          v-on:load="onLoadIframe"
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
import { Eth } from "web3-eth";
import { Contract } from "web3-eth-contract";


export default Vue.extend({
  name: "Home",
  data: () =>(
    {
      CONTRACT_ADDRESS: "0x75BCc6456812A005084391ADfBB21c6C54726db5",
      iframe_created: false,
      web3: {} as Web3,
      eternalStorage: {} as Contract,
      eternalStorageJson: require("./../components/EternalStorage.json")
    }
  ),
  methods: {
    receiveMessage(event : any) {
      if (event.data === "idle") {
        console.log("POC communication between iframe and parent");

      }
    },

    onLoadIframe() {
      console.log("iframe created");
      this.iframe_created = true;
      const lat = this.$route.query.lat;
      const lng = this.$route.query.lng;
      if (lat != null && lng != null) {
        this.sendDataToIframe({event: "setCenter", data: {lat: lat, lng: lng}});
      }
      this.getHeatMapData(120, "0x0").then(
        data => {
          this.sendDataToIframe({event: "sendData", data: data});
        }
      );
    },
    sendDataToIframe(data: {event: string, data: any}) {
      if (this.iframe_created) {
        const iframe = document.querySelector("iframe");
        if (iframe != null && iframe.contentWindow != null) {
          iframe.contentWindow.postMessage({
            "event": data.event,
            "data": data.data
          }, "*");
        }
      }
      else {
        console.log("iframe not created, cant send message");
      }
    },

    getHeatMapData(timestamp : number,  coords : string) : Promise<any>{
      console.log("called");
      return new Promise<any>(
        resolve => {
          this.eternalStorage.methods.getHashValue(timestamp, coords).call().then(
        function(multihash : Multihash){
          let ipfs_hash = ipfs_utils.decode(multihash);
          console.log(ipfs_hash);
          fetch("https://cf-ipfs.com/ipfs/" + ipfs_hash, {
            headers: {
              "accept": "application/json",
            }
          }).then(
            response => response.json().then(
              data => {
                //console.log(data)
                resolve(data);
              }
            )
          );
        }
      );
        }
      )
    }
  },
  mounted () {
    window.addEventListener("message", this.receiveMessage);
  },
  beforeDestroy () {
    window.removeEventListener("message", this.receiveMessage);
  },
  created(){
    this.web3 = new Web3(
      new Web3.providers.HttpProvider(
        `https://goerli.infura.io/v3/4fb5537dd7124137a2bc95668e973d76`
      )
    );
    this.eternalStorage = new this.web3.eth.Contract(this.eternalStorageJson.abi, this.CONTRACT_ADDRESS);
  }
});
</script>
