import Vue from "vue";
import VueRouter, {RouteConfig} from "vue-router";
import Home from "@/views/Home.vue";
import About from "@/views/About.vue";
import Clusters from "@/views/Clusters.vue";

Vue.use(VueRouter);

const routes: Array<RouteConfig> = [
  {
    path: "/",
    redirect: "/home",
  },
  {
    path: "/home",
    component: Home,
  },
  {
    path: "/about",
    component: About,
  },
  {
    path: "/clusters",
    component: Clusters,
  },
];

const router = new VueRouter(
  {
    mode: "history",
    routes,
  }
);

export default router;
