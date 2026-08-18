import { defineConfig } from "vite";

export default defineConfig({
  // The Space serves this build under /browser/, while `npm run dev` serves it
  // at /. A relative base makes the emitted asset URLs resolve against the page
  // instead of the origin root, so one build works at either mount point.
  base: "./",
});
