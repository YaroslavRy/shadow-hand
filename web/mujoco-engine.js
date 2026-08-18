import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";

// Resolved against the page rather than the origin root: the Space serves this
// build under /browser/, `npm run dev` serves it at /.
const SCENE_ROOT = new URL("mujoco/shadow_hand/", document.baseURI);
const WORK_ROOT = "/working/shadow_hand";
const FILES = [
  "my_scene.xml", "right_hand.xml",
  "assets/f_distal_pst.obj", "assets/f_knuckle.obj", "assets/f_middle.obj",
  "assets/f_proximal.obj", "assets/forearm_0.obj", "assets/forearm_1.obj",
  "assets/forearm_collision.obj", "assets/lf_metacarpal.obj",
  "assets/mounting_plate.obj", "assets/palm.obj", "assets/th_distal_pst.obj",
  "assets/th_middle.obj", "assets/th_proximal.obj", "assets/wrist.obj",
];

async function fetchAsset(path) {
  const response = await fetch(new URL(path, SCENE_ROOT));
  if (!response.ok) throw new Error(`asset fetch failed: ${path}`);
  return path.endsWith(".xml") ? await response.text() : new Uint8Array(await response.arrayBuffer());
}

export async function createShadowHandEngine() {
  const progress = typeof window === "undefined" ? null : document.querySelector("#engine-status");
  if (progress) progress.textContent = "Loading MuJoCo WASM…";
  const mujoco = await loadMujoco({ locateFile: () => mujocoWasmUrl });
  if (progress) progress.textContent = "Loading Menagerie XML and meshes…";
  mujoco.FS.mkdir("/working");
  mujoco.FS.mkdir(WORK_ROOT);
  mujoco.FS.mkdir(`${WORK_ROOT}/assets`);
  await Promise.all(FILES.map(async (path) => {
    mujoco.FS.writeFile(`${WORK_ROOT}/${path}`, await fetchAsset(path));
  }));
  // @mujoco/mujoco 3.11 exposes XML loading on MjModel, not the module root.
  // Loading from the virtual filesystem keeps <include> and meshdir resolution intact.
  if (progress) progress.textContent = "Compiling real Shadow Hand MJCF…";
  const model = mujoco.MjModel.from_xml_path(`${WORK_ROOT}/my_scene.xml`);
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);
  return { mujoco, model, data };
}
