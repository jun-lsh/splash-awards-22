<template>
  <v-container fluid class="px-12 py-6">
    <h1>Clusters</h1>

    <div style="max-width: 80rem">
      <v-row class="mt-4">
        <v-col cols="12" lg="4" md="8" v-for="item in sortedClusters" :key="item.id">
          <router-link
            class="unit-card"
            :to="'/home?lat=' + item.lat + '&lng=' + item.lng"
            style="text-decoration: none; color: inherit;"
          >
            <v-card class="pa-4 fill-height d-flex flex-column" elevation="2">
              <v-card-title>#{{ item.id }}</v-card-title>
              <v-card-text>
                Lat, Long: {{ item.lat }}, {{ item.lng }}<br/>
                Size: {{ item.conc }}
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
export default Vue.extend({
  name: "Clusters",
  computed: {
    clusters(): Array<{
      id: number,
      lat: number,
      lng: number,
      conc : number
    }> {
      return [
        {
          id: 1,
          lat: 10,
          lng: 15,
          conc: 100,
        },
      ];
    },
    sortedClusters(): Array<{
      id: number,
      lat: number,
      lng: number,
      conc : number
    }> {
        let clusters = this.clusters;
        return clusters.sort((a,b) => a.conc - b.conc).slice(0,10);
    }
  }
});

</script>

<style scoped>

</style>
