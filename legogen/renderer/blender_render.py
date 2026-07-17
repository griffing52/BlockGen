"""Runs INSIDE Blender to render LDraw models to PNG tiles via the ImportLDraw addon.

    blender --background --python legogen/renderer/blender_render.py -- <job.json>

The job JSON is::

    {
      "ldraw_lib": "data/lego/ldraw/ldraw",   # LDraw parts library root
      "res": 512, "samples": 48, "engine": "CYCLES",
      "azim": 40, "elev": 28, "margin": 1.12,
      "colour_scheme": "lgeo", "look": "normal",
      "tiles": [{"src": "…/10015-1.mpd", "out": "…/tiles/10015-1.png"}, …]
    }

Each tile is rendered with a fixed orthographic isometric camera framed to the
model's bounding box, on a transparent background, so the tiles compose into a
clean uniform grid (the Minecraft ``renderer/grid.py`` aesthetic). Per-model
failures are caught and logged so one bad file never aborts the batch.
"""

import json
import math
import sys
import traceback

import bpy
import mathutils


def _argv_after_ddash():
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def enable_addon():
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_importldraw")
    except Exception:  # noqa: BLE001
        pass


def clear_scene():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Purge orphaned meshes/materials so a long batch doesn't leak memory.
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True,
                                           do_recursive=True)
        except Exception:  # noqa: BLE001
            break


def configure_gpu(scene):
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.type != "CPU" for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type != "CPU" or True
                    scene.cycles.device = "GPU"
                    return backend
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return "CPU"


def setup_render(job):
    scene = bpy.context.scene
    engine = job["engine"]
    scene.render.engine = "BLENDER_EEVEE_NEXT" if engine == "EEVEE" else "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = job["res"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = job["samples"]
        scene.cycles.use_denoising = True
        backend = configure_gpu(scene)
        print(f"[legogen] Cycles device: {scene.cycles.device} ({backend})", flush=True)
    else:
        scene.eevee.taa_render_samples = job["samples"]


def setup_world():
    world = bpy.data.worlds.new("legogen_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 0.75  # soft even fill so shadowed faces stay legible


def add_key_light(azim_deg, elev_deg):
    data = bpy.data.lights.new("legogen_sun", type="SUN")
    data.energy = 3.5
    data.angle = math.radians(3.0)
    obj = bpy.data.objects.new("legogen_sun", data)
    bpy.context.collection.objects.link(obj)
    # Aim the sun from over the camera's shoulder.
    az = math.radians(azim_deg + 25)
    el = math.radians(elev_deg + 25)
    direction = mathutils.Vector((math.cos(el) * math.cos(az),
                                  math.cos(el) * math.sin(az), math.sin(el)))
    obj.rotation_euler = (-direction).to_track_quat("Z", "Y").to_euler()


def scene_bbox():
    """World-space bounding box over all mesh objects."""
    mins = mathutils.Vector((1e18, 1e18, 1e18))
    maxs = mathutils.Vector((-1e18, -1e18, -1e18))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                mins[i] = min(mins[i], world[i])
                maxs[i] = max(maxs[i], world[i])
    if not found:
        return None, None
    return mins, maxs


def frame_ortho_camera(azim_deg, elev_deg, margin):
    mins, maxs = scene_bbox()
    if mins is None:
        return None
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).length * 0.5, 1e-3)

    cam_data = bpy.data.cameras.new("legogen_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0 * radius * margin
    cam = bpy.data.objects.new("legogen_cam", cam_data)
    bpy.context.collection.objects.link(cam)

    az, el = math.radians(azim_deg), math.radians(elev_deg)
    view_dir = mathutils.Vector((math.cos(el) * math.cos(az),
                                 math.cos(el) * math.sin(az), math.sin(el)))
    cam.location = center + view_dir * (radius * 4.0)
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.clip_start = 0.01
    cam_data.clip_end = radius * 12.0
    bpy.context.scene.camera = cam
    return cam


def render_one(job, tile):
    clear_scene()
    bpy.ops.import_scene.importldraw(
        filepath=tile["src"],
        ldrawPath=job["ldraw_lib"],
        look=job["look"],
        colourScheme=job["colour_scheme"],
        addEnvironment=False,
        importCameras=False,
        positionCamera=False,
        positionOnGround=True,
        useLogoStuds=False,
        instanceStuds=False,
        bevelEdges=True,
        linkParts=True,
        useUnofficialParts=True,
    )
    setup_world()
    add_key_light(job["azim"], job["elev"])
    if frame_ortho_camera(job["azim"], job["elev"], job["margin"]) is None:
        raise RuntimeError("no mesh geometry imported")
    bpy.context.scene.render.filepath = tile["out"]
    bpy.ops.render.render(write_still=True)


def main():
    args = _argv_after_ddash()
    if not args:
        print("[legogen] no job json passed after --", flush=True)
        return
    job = json.loads(open(args[0]).read())
    enable_addon()
    setup_render(job)
    ok = 0
    for i, tile in enumerate(job["tiles"]):
        try:
            render_one(job, tile)
            ok += 1
            print(f"[legogen] {i + 1}/{len(job['tiles'])} OK {tile['out']}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[legogen] {i + 1}/{len(job['tiles'])} FAIL {tile['src']}", flush=True)
            print(traceback.format_exc(), flush=True)
    print(f"[legogen] DONE {ok}/{len(job['tiles'])} rendered", flush=True)


if __name__ == "__main__":
    main()
