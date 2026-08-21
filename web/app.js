import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { createShadowHandEngine } from "./mujoco-engine.js";
import { SCENE_PARAMS } from "./params.js";
import { extractTargets, smoothTargets, mirrorLandmarks, SMOOTHING_FACTOR, PHYSICS_TIMESTEP } from "./retarget.js";

// Pin the wasm runtime to the installed @mediapipe/tasks-vision version (0.10.21);
// a mismatch between the JS API and the wasm backend breaks inference.
const resolveVisionFiles = FilesetResolver.forVisionTasks.bind(FilesetResolver);
FilesetResolver.forVisionTasks = () => resolveVisionFiles("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm");

const video = document.querySelector("#video"), canvas = document.querySelector("#overlay"), ctx = canvas.getContext("2d");
const status = document.querySelector("#status"), engineStatus = document.querySelector("#engine-status"), root = document.querySelector("#view");
// my_scene.xml puts the MuJoCo ground plane at z=-0.2, and Three's Y is
// MuJoCo's Z, so the visual floor has to sit there or the hand looks unmoored.
const FLOOR_Y = SCENE_PARAMS.floorY;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x03070d);
const camera = new THREE.PerspectiveCamera(SCENE_PARAMS.camera.fov, 1, SCENE_PARAMS.camera.near, SCENE_PARAMS.camera.far); camera.position.fromArray(SCENE_PARAMS.camera.initialPosition);
const renderer = new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; root.append(renderer.domElement);
const orbit = new OrbitControls(camera, renderer.domElement); orbit.target.fromArray(SCENE_PARAMS.orbit.initialTarget); orbit.minDistance=SCENE_PARAMS.orbit.minDistance; orbit.maxDistance=SCENE_PARAMS.orbit.maxDistance; orbit.update();
scene.add(new THREE.HemisphereLight(0xaad8ff, 0x03070d, 2));
const light = new THREE.DirectionalLight(0xdfffff, 3); light.position.set(.3, .6, .4); light.castShadow = true; scene.add(light);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(4,4), new THREE.MeshStandardMaterial({color:0x101b2b})); floor.rotation.x = -Math.PI/2; floor.position.y = FLOOR_Y; floor.receiveShadow = true; scene.add(floor);
new ResizeObserver(() => { const r=root.getBoundingClientRect(); renderer.setSize(r.width,r.height,false); camera.aspect=r.width/r.height; camera.updateProjectionMatrix(); }).observe(root);

const material = new THREE.MeshStandardMaterial({color:0x7f9eb5,metalness:.8,roughness:.25});
let wasmEngine, realHand;

// Render MuJoCo's compiled visual geoms directly, but place each geom from
// MuJoCo's world-space geom transform instead of rebuilding the hierarchy here.
function createRealHand(engine) {
  const {model}=engine, group=new THREE.Group(), nodes=new Map(), geometries=new Map(), matrix=new THREE.Matrix4(), convert=new THREE.Matrix4().makeRotationX(-Math.PI/2); group.updateMatrixWorld(true);
  for(let id=0;id<model.ngeom;id+=1){
    const geom=model.geom(id);
    if(Number(geom.group)!==2||Number(geom.dataid)<0)continue;
    const meshId=Number(geom.dataid);
    let geometry=geometries.get(meshId);
    if(!geometry){
      const mesh=model.mesh(meshId),va=Number(mesh.vertadr),vn=Number(mesh.vertnum),fa=Number(mesh.faceadr),fn=Number(mesh.facenum);
      geometry=new THREE.BufferGeometry();
      geometry.setAttribute("position",new THREE.BufferAttribute(model.mesh_vert.slice(va*3,(va+vn)*3),3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(model.mesh_face.slice(fa*3,(fa+fn)*3)),1));
      geometry.computeVertexNormals();
      geometries.set(meshId,geometry);
    }
    const visual=new THREE.Mesh(geometry,material);
    visual.matrixAutoUpdate=false;
    visual.castShadow=true;
    visual.receiveShadow=true;
    nodes.set(id,visual);
    group.add(visual);
  }
  return {
    group,
    count:nodes.size,
    sync(){
      nodes.forEach((node,id)=>{
        const geom=engine.data.geom(id),m=geom.xmat,p=geom.xpos;
        // Matrix4.set is row-major, matching MuJoCo's row-major xmat, and the
        // translation goes in the fourth column.
        matrix.set(m[0],m[1],m[2],p[0],m[3],m[4],m[5],p[1],m[6],m[7],m[8],p[2],0,0,0,1);
        // Vertices stay in MuJoCo's Z-up local frame, so only the world
        // transform is rebased into Three's Y-up frame: no trailing inverse.
        node.matrix.copy(convert).multiply(matrix);
      });
    }
  };
}
function setControls(values) { if(!wasmEngine)return; for(const [name,value] of Object.entries(values)) wasmEngine.data.actuator(name).ctrl=value; }

// Python steps physics every loop iteration, independent of whether a hand was
// seen, and at a fixed 16 steps per 30 FPS frame — i.e. real time. Render here
// is 60 FPS and tracking 30 Hz, so drive the sim off the wall clock
// instead of the frame count or the dynamics run at the wrong rate.
let simCarry = 0;
function stepPhysics(elapsedSeconds) {
  if(!wasmEngine) return;
  simCarry = Math.min(simCarry + elapsedSeconds, .25);
  let steps = Math.floor(simCarry / PHYSICS_TIMESTEP);
  simCarry -= steps * PHYSICS_TIMESTEP;
  while(steps-- > 0) wasmEngine.mujoco.mj_step(wasmEngine.model, wasmEngine.data);
  realHand?.sync();
}
// Frame the whole rendered model, mount included — deriving the span from the
// palm and fingertips alone cropped the forearm base out of the bottom.
function frameLoadedHand() {
  if(!realHand) return;
  const box = new THREE.Box3().setFromObject(realHand.group);
  if(box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  center.add(new THREE.Vector3(...SCENE_PARAMS.framing.centerOffset));
  const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) / SCENE_PARAMS.framing.fovDivisor) * SCENE_PARAMS.framing.distanceScale;
  orbit.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(...SCENE_PARAMS.framing.viewDirection).normalize().multiplyScalar(distance));
  camera.near = Math.max(.01, distance - radius * SCENE_PARAMS.framing.nearRadiusScale);
  camera.far = distance + radius * SCENE_PARAMS.framing.farRadiusScale;
  camera.updateProjectionMatrix();
  orbit.update();
}
createShadowHandEngine().then(engine=>{wasmEngine=engine;realHand=createRealHand(engine);scene.add(realHand.group);for(let i=0;i<30;i++)engine.mujoco.mj_step(engine.model,engine.data);realHand.sync();frameLoadedHand();engineStatus.textContent=`Real MuJoCo WASM · nq ${engine.model.nq} · nu ${engine.model.nu} · ${realHand.count} visible geoms`;}).catch(error=>{console.error("MuJoCo WASM failed",error);engineStatus.textContent=`MuJoCo WASM error: ${error.message}`;});

const links=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
function draw(p){canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle="#63d7ca";ctx.fillStyle="white";ctx.lineWidth=3;links.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(p[a].x*canvas.width,p[a].y*canvas.height);ctx.lineTo(p[b].x*canvas.width,p[b].y*canvas.height);ctx.stroke();});p.forEach(q=>{ctx.beginPath();ctx.arc(q.x*canvas.width,q.y*canvas.height,4,0,7);ctx.fill();});}
let tracker,last=-1,lastInference=0,count=0,t0=performance.now();
// The tracker is assigned before getUserMedia resolves, so the render loop can
// reach this while the video still has no frame. Feeding MediaPipe a 0x0 frame
// fails its ImageToTensor ROI check and permanently breaks the graph, so the
// video must be confirmed to have real dimensions before any inference.
let previousTargets={},lastFrame=performance.now();
function animate(){
  requestAnimationFrame(animate);
  const now=performance.now();
  stepPhysics((now-lastFrame)/1000);
  lastFrame=now;
  orbit.update();
  renderer.render(scene,camera);
  count++;
  if(now-t0>1000){document.querySelector("#fps").textContent=`${count} render FPS · 30 Hz tracking`;count=0;t0=now;}
  if(!tracker||video.readyState<2||!video.videoWidth||!video.videoHeight||video.currentTime===last||now-lastInference<33)return;
  last=video.currentTime;lastInference=now;
  let detected;
  try{detected=tracker.detectForVideo(video,now);}
  catch(error){console.error("detectForVideo failed",error);status.textContent=`Tracker error: ${error?.message||String(error)}`;return;}
  const p=detected.landmarks?.[0];
  if(!p){ctx.clearRect(0,0,canvas.width,canvas.height);status.textContent="No hand in frame — hold your whole hand up to the camera";return;}
  draw(p);
  previousTargets=smoothTargets(extractTargets(mirrorLandmarks(p)),previousTargets,SMOOTHING_FACTOR);
  setControls(previousTargets);
  status.textContent="Hand detected · controls drive real MuJoCo";
}
// The 3D view is independent of the tracker: orbiting and the loaded hand pose
// must be visible before (and without) the camera ever being enabled.
animate();
document.querySelector("#start").onclick=async event=>{event.target.textContent="Loading tracker…";try{const vision=await FilesetResolver.forVisionTasks();event.target.textContent="Loading hand model…";tracker=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",delegate:"GPU"},runningMode:"VIDEO",numHands:1,// Match tracking.py's mp.solutions.hands: detection 0.7, tracking 0.5.
minHandDetectionConfidence:.7,minHandPresenceConfidence:.5,minTrackingConfidence:.5});event.target.textContent="Requesting camera…";if(!navigator.mediaDevices?.getUserMedia)throw new Error("getUserMedia is unavailable in this browser context. Open the app in a normal browser tab via https, http://127.0.0.1, or http://localhost and allow camera access.");video.srcObject=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480},frameRate:{ideal:30}},audio:false});await video.play();event.target.remove();status.textContent="Waiting for hand…";}catch(error){console.error("Camera setup failed",error);event.target.textContent="Retry camera";status.textContent=`Camera error: ${error?.message || String(error)}`;}};
