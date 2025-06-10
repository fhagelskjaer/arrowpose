import numpy as np
import trimesh
import pyrender
import cv2

class DepthCheck:
    def __init__(self, model_name_def, w, h, fx, fy, cx, cy):
        fuze_trimesh = trimesh.load( model_name_def )
        fuze_trimesh = fuze_trimesh.apply_scale(0.001)

        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, smooth=False)

        self.r = pyrender.OffscreenRenderer(w, h)

        self.scene = pyrender.Scene()
        self.scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
        )

        self.nc = pyrender.Node(camera=camera)
        self.scene.add_node(self.nc)

    def compute(self, object_detection, orig_depth, back_dist, acc_dist):
        x_rot = np.array([
          [1.0, 0,   0,   0.0],
          [0.0,  -1.0, 0.0, 0.0],
          [0.0,  0,   -1.0,   0.0],
          [0.0,  0.0, 0.0, 1.0],
        ])

        # Read object pose and set camera
        obj_pose = object_detection.copy() #
        obj_pose[:3,3] *= 0.001
        camera_pose = np.dot(np.linalg.inv(obj_pose), x_rot)

        # Update camera position
        self.scene.set_pose(self.nc, pose=camera_pose)

        # Render the scene
        synth_depth = self.r.render(self.scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)

        # First calculate object mask from synth depth image
        depth_mask_orig = np.float32(synth_depth != 0)
        kernel = np.ones((3,3), np.uint8) # TODO

        depth_mask = np.where( ( synth_depth - (orig_depth/1000.0)) < back_dist, depth_mask_orig, np.zeros_like(depth_mask_orig) )
        # depth_mask_edge = cv2.dilate(depth_mask, kernel)
        depth_mask = cv2.erode(depth_mask, kernel)

        # Calculate the depth comparison
        depth_count = np.where( depth_mask == 1, np.abs(np.float32(orig_depth)/1000.0 - np.float32(synth_depth)) < acc_dist, np.zeros_like(depth_mask) )
        return np.sum(depth_count)/np.sum(depth_mask)
