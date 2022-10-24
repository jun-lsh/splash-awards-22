<template>
  <v-container fluid>

    <v-row class="pa-6">
      <v-col cols="2"/>
      <v-col cols="4" class="d-flex justify-center" style="font-size: larger;">
        <div style="justify-content: center">
          <h2 style="text-align: center">What are microplastics?</h2> <br/>
          <p style="text-align: justify; text-justify: inter-word;">
            Microplastics are tiny plastic particles less than five millimeters in diameter. There
            are two categories of microplastics, primary microplastics and secondary microplastics.
            Microplastics are shaping into one of the worst environmental crises of our time, yet
            not enough research is being done.
          </p>
        </div>
      </v-col>

      <v-col cols="6" class="d-flex justify-center">
        <v-card>
          <v-img src="images/primary_and_secondary_plastics.png" alt="Diagram explaining primary and secondary microplastics"/>
          <v-card-text>Secondary microplastics make up a much larger proportion of microplastics</v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-row class="pa-6">
      <v-col cols="4" class="d-flex justify-center">
        <v-card>
          <v-img src="images/nurdles.png" alt="Image of a handful of nurdles"/>
          <v-card-text>Nurdles are the building blocks of plastic products, and 230,000 tonnes of nurdles are dumped into the ocean every year </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="8" class="d-flex justify-center">
        <v-row>
          <v-col cols="12" style="font-size: larger">
            <div style="justify-content: center">
              <p style="text-align: justify; text-justify: inter-word;">
                Primary microplastics are plastic materials intentionally designed to be small, as
                part of commercial or industrial use. These primary microplastics can arise in the
                form of beauty or cosmetic products, plastic fabric, or nurdles which are the small
                raw materials to produce plastic in industrial settings.
              </p>
            </div>
          </v-col>

          <v-col cols="12">
            <v-row>
              <v-col cols="4"/>
              <v-col cols="6" class="d-flex justify-center" style="font-size: larger">
                <div style="justify-content: center">
                  <p style="text-align: justify; text-justify: inter-word;">
                    Secondary microplastics come from the breakdown of normal plastic, such as plastic bottles and bags. The breakdown can be caused by wind abrasion, radiation from the sun, and ocean currents
                  </p>
                </div>
              </v-col>

              <v-col cols="2" class="d-flex justify-center">
                <v-card outlined color="transparent">
                  <v-img class="rounded-lg" src="images/plastic_bottle.png" alt="Image of a water bottle in the sea breaking down into microplastics"/>
                </v-card>
              </v-col>
            </v-row>

          </v-col>
        </v-row>
      </v-col>
    </v-row>

    <div style="text-align: center; width: 50%; margin: 0 auto">
      <b>How can I help out?</b> <br/>
      <p>
        Coastal cleanup efforts can mitigate the risks that plastic pollution brings to the animals that live on the beach.
        You can read more about coastal cleanup efforts or organise one in Singapore using the links below.
        Read up on the appropriate preparations and gather a group of people to host a beach clean up!
      </p>
      <a href="https://oceanconservancy.org/trash-free-seas/international-coastal-cleanup/">https://oceanconservancy.org/trash-free-seas/international-coastal-cleanup/</a> <br/>
      <a href="https://www.publichygienecouncil.sg/beachcleanup">https://www.publichygienecouncil.sg/beachcleanup</a> <br/> <br/>

      <p>
        There are also many groups focusing on cleaning up the large garbage patches in the ocean,
        While we are unable to go out there and clean it up ourselves, we can donate to these organisations
        that focuses on making our oceans a cleaner place.
      </p>
      <a href="http://www.oceansplasticleanup.com/">http://www.oceansplasticleanup.com/</a> <br/>
      <a href="https://teamseas.org/">https://teamseas.org/</a> <br/>
    </div>
  </v-container>
</template>

<script lang="ts">
import Vue from "vue";

export default Vue.extend({
  name: "About"
});
</script>

<style scoped>

</style>
