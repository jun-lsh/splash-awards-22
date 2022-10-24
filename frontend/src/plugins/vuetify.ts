import Vue from "vue";
import Vuetify from "vuetify";
import telegram from "@/components/telegram.vue";

Vue.use(Vuetify);

export default new Vuetify({
  icons: {
    values: {
      telegram: {
        component: telegram,
      },
    },
  },
});
