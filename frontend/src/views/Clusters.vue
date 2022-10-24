<template>
  <v-container fluid class="px-12 py-6">
    <h1>Top 10 largest clusters</h1>

    <div style="max-width: 80rem">
      <v-row class="mt-4">
        <v-col cols="12" lg="4" md="8" v-for="(item, index) in clusters" :key="index">
          <router-link
            class="unit-card"
            :to="'/home?lat=' + item.lat + '&lng=' + item.lng"
            style="text-decoration: none; color: inherit;"
          >
            <v-card class="pa-4 fill-height d-flex flex-column" elevation="2">
              <v-card-title>Concentration: {{ Math.round(item.conc*100)/100 }}</v-card-title>
              <v-card-text>
                Lat, Long: {{ Math.round(item.lat*1000)/1000 }}, {{ Math.round(item.lng*1000)/1000 }}<br/>
              </v-card-text>
            </v-card>
          </router-link>
        </v-col>
      </v-row>
    </div>
  </v-container>
</template>

<script lang="ts">
import Vue from "vue";
import {Multihash} from "@/types/multihash";
import ipfs_utils from "@/plugins/ipfs_decode";
import Web3 from "web3";
import {Contract} from "web3-eth-contract";
export default Vue.extend({
  name: "Clusters",
  data: () =>(
    {
      CONTRACT_ADDRESS: "0x75BCc6456812A005084391ADfBB21c6C54726db5",
      iframe_created: false,
      web3: {} as Web3,
      eternalStorage: {} as Contract,
      eternalStorageJson: require("./../components/EternalStorage.json"),
      clusters: {} as Array<{lat: number, lng: number, conc: number}>,
    }
  ),
  methods: {
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
      );
    }
  },
  mounted(){
    this.web3 = new Web3(
      new Web3.providers.HttpProvider(
        `https://goerli.infura.io/v3/4fb5537dd7124137a2bc95668e973d76`
      )
    );
    this.eternalStorage = new this.web3.eth.Contract(this.eternalStorageJson.abi, this.CONTRACT_ADDRESS);

    this.getHeatMapData(120, "0x0").then(data => {
      this.clusters = data.data;
      this.clusters = this.clusters.sort((a, b) => b.conc - a.conc).slice(0, 10);
    });
  }
});

</script>

<style scoped>

</style>
