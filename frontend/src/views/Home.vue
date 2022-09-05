<template>
  <v-container fluid class="px-12 py-6 fill-height">
    <v-row class="d-flex align-center justify-center fill-height">
      <v-col cols="6" style="height: 100%">
        <!-- <iframe
          id="leaflet"
          src="leaflet.html"
          frameborder="0"
          width="100%"
          height="75%"
        /> -->

        <l-map style="height: 600px" :zoom="zoom" :center="center">
          <l-tile-layer :url="url" :attribution="attribution"></l-tile-layer>
          <l-marker v-for="marker in centroids" :key="marker.count" :lat-lng="marker"></l-marker>
        </l-map>
      </v-col>
    </v-row>

  </v-container>
</template>

<script lang="ts">
import Vue from "vue";
import Web3 from "web3";
import ipfs_utils from "@/plugins/ipfs_decode";
import { Multihash } from "@/types/multihash";


import { latLng } from "leaflet";
import { LMap, LTileLayer, LMarker, LPopup, LTooltip } from "vue2-leaflet";
import { Icon } from 'leaflet';

//delete Icon.Default.prototype._getIconUrl;
Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

export default Vue.extend({
  name: "Home",
  components: {
    LMap,
    LTileLayer,
    LMarker,
  },
  data: () =>(
    {
      CONTRACT_ADDRESS: "0x3260Df12C458Ac84CBbeFb82F92E8Ddc57927CD7",
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution:
        'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
      zoom: 3,
      center: [14.5994, 28.6731],
      centroids: [
    { lat: -62.68893, lng: 27.52887, count: 1 },
    { lat: -62.69893, lng: 27.52687, count: 1 },
    { lat: -62.58893, lng: 27.52887, count: 1 },
    { lat: -62.69993, lng: 26.52887, count: 1 },
    { lat: -62.68833, lng: 27.57887, count: 1 }]
    }
  ),
  mounted(){
  },
  methods:{
  },
		
  created(){
    const web3 = new Web3(
      new Web3.providers.HttpProvider(
        `https://ropsten.infura.io/v3/${process.env.VUE_APP_INFURA_API_KEY}`
    )
    );
    const eternalStorageJson = require("./../../../truffle/build/contracts/EternalStorage.json");
    const eternalStorage = new web3.eth.Contract(eternalStorageJson.abi, this.CONTRACT_ADDRESS);
    console.log("called")
    const value = "0x0";
    eternalStorage.methods.getHashValue(1, value).call().then(
      function(multihash : Multihash){
        let ipfs_hash = ipfs_utils.decode(multihash);
        console.log(ipfs_hash)
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
        )
      }
    );
  }
});
</script>
