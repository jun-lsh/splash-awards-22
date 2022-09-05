<template>
  <v-container fluid class="px-12 py-6">
    <router-link to="/">
      &lt; Back
    </router-link>
    <h1>404 Not found, double check the link you went to</h1>
  </v-container>
</template>

<script lang="ts">
import Vue from "vue";

export default Vue.extend({
  name: "NotFound",
});
</script>

<style lang="scss" scoped>
</style>
