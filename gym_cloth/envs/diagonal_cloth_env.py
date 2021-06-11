"""
An OpenAI Gym-style environment for the cloth smoothing experiments. It's not
exactly their interface because we pass in a configuration file. See README.md
document in this directory for details.
"""
import pyximport; pyximport.install()

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
from os.path import join
import subprocess
import sys
import time
import datetime
import logging
import json
import yaml
import subprocess
import trimesh
import cv2
import datetime
import pickle
import copy
import pkg_resources
from gym_cloth.physics.cloth import Cloth
from gym_cloth.physics.point import Point
from gym_cloth.physics.gripper import Gripper
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.spatial import ConvexHull

_logging_setup_table = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
}

# Thresholds for successful episode completion.
# Keys are possible values for reward_type in config.
_REWARD_THRESHOLDS = {
    'coverage': 0.92,
    'coverage-delta': 0.92,
    'height': 0.85,
    'height-delta': 0.85,
    'variance': 2,
    'variance-delta': 2,
    'folding': 0.01
}

_EPS = 1e-5


class DiagonalClothEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, subrank=None, start_state_path=None):
        """Various initialization for the environment.

        Not to be confused with the initialization for the _cloth_.

        See the following for how to use proper seeding in gym:
          https://github.com/openai/gym/blob/master/gym/utils/seeding.py
          https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py
          https://stackoverflow.com/questions/5836335/
          https://stackoverflow.com/questions/22994423/

        The bounds of (1,1,1) are for convenience and should be the same bounds
        as the cloth has internally.

        RL algorithms refer to observation_space and action_space when building
        neural networks.  Also, we may need to sample from our action space for
        a random policy.  For the actions, we should enforce clipping, but it's
        a subtle issue. See how others do it for MuJoCo.

        Optional:
        - `subrank` if we have multiple envs in parallel, to make it easy to
          tell which env corresponds to certain loggers.
        - `start_state_path` if we want to force cloth to start at a specific
          state.  Represents the path to the state file.
        """
        # NOTE! This is the only time we load from the yaml file. We do this
        # when ClothEnv is called, and it uses the fixed values from the yaml
        # file. Changing the yaml file later while code is running is OK as it
        # will not affect results.
        with open(cfg_file, 'r') as fh:
            cfg = yaml.safe_load(fh)
        self.cfg              = cfg
        self.cfg_file         = cfg_file
        self.max_actions      = cfg['env']['max_actions']
        self.grip_radius      = cfg['env']['grip_radius']
        self.render_gl        = cfg['init']['render_opengl']
        self._init_type       = cfg['init']['type']
        self._clip_act_space  = cfg['env']['clip_act_space']
        self._delta_actions   = cfg['env']['delta_actions']
        self._obs_type        = cfg['env']['obs_type']
        self._oracle_reveal   = cfg['env']['oracle_reveal']
        self._use_depth       = cfg['env']['use_depth']
        self._use_rgbd        = cfg['env']['use_rgbd']
        self.bounds = bounds  = (1, 1, 1)
        self.render_proc      = None
        self.render_port      = 5556
        self._logger_idx      = subrank
        self._occlusion_vec   = [True, True, True, True]
        self.__add_dom_rand   = cfg['env']['use_dom_rand']              # string
        self._add_dom_rand    = (self.__add_dom_rand.lower() == 'true') # boolean
        self.dom_rand_params  = {} # Ryan: store domain randomization params to keep constant per episode

        if start_state_path is not None:
            with open(start_state_path, 'rb') as fh:
                self._start_state = pickle.load(fh)
        else:
            self._start_state = None

        # Reward design. Very tricky. BE EXTRA CAREFUL WHEN ADJUSTING,
        # particularly if we do DeepRL with demos, because demos need rewards
        # consistent with those seen during training.
        self.reward_type = cfg['env']['reward_type']

        # Create observation ('1d', '3d') and action spaces. Possibly make the
        # obs_type and other stuff user-specified parameters.
        self._slack = 0.25
        self.num_w = num_w = cfg['cloth']['num_width_points']
        self.num_h = num_h = cfg['cloth']['num_height_points']
        self.num_points = num_w * num_h
        lim = 100
        if self._obs_type == '1d':
            self.obslow  = np.ones((3 * self.num_points,)) * -lim
            self.obshigh = np.ones((3 * self.num_points,)) * lim
            self.observation_space = spaces.Box(self.obslow, self.obshigh)
        elif self._obs_type == 'blender':
            self._hd = 224
            self._wd = 224
            self.obslow = np.zeros((self._hd, self._wd, 3)).astype(np.uint8)
            self.obshigh = np.ones((self._hd, self._wd, 3)).astype(np.uint8)
            self.observation_space = spaces.Box(self.obslow, self.obshigh, dtype=np.uint8)
        else:
            raise ValueError(self._obs_type)

        if self._clip_act_space:
            # Applies regardless of 'delta actions' vs non deltas.  Note misleading
            # 'clip' name, because (0,1) x/y-coords are *expanded* to (-1,1).
            self.action_space = spaces.Box(
                low= np.array([-1., -1., -1.]),
                high=np.array([ 1.,  1.,  1.])
            )
        else:
            self.action_space = spaces.Box(
                low= np.array([-1., -1., -1., -1.]),
                high=np.array([1., 1.,  1.,  1.])
            )


        # Bells and whistles
        self._setup_logger()
        self.seed()
        self.debug_viz = cfg['init']['debug_matplotlib']

    @property
    def state(self):
        """Returns state representation as a numpy array.

        The heavy duty part of this happens if we're using image observations,
        in which we form the mesh needed for Blender to render it. We also pass
        relevant arguments to the Blender script.
        """
        if self._obs_type == '1d':
            lst = []
            for pt in self.cloth.pts:
                lst.extend([pt.x, pt.y, pt.z])
            return np.array(lst)
        elif self._obs_type == 'blender':
            # Ryan: implement RGBD
            if self._use_rgbd == 'True':
                img_rgb = self.get_blender_rep('False')
                img_d = self.get_blender_rep('True')[:,:,0]
                return np.dstack((img_rgb, img_d))
            else:
                return self.get_blender_rep(self._use_depth)
        else:
            raise ValueError(self._obs_type)

    @property
    def obs_1d(self):
        """ 3D Observation X,Y,Z of the points"""
        lst = []
        for pt in self.cloth.pts:
            lst.append(np.array([pt.x, pt.y, pt.z]))
        return np.array(lst)

    @property
    def grasped_pts(self):
        """ Index of grasped points the the self.obs_1d array"""
        cloth_points = self.obs_1d
        idx_grasped_list = []
        grasp_points = []
        for pt in self.gripper.grabbed_pts:
            grasp_points.append(np.array([pt.x, pt.y, pt.z]))
            pt_list = np.array([pt.x, pt.y, pt.z]).tolist()
            if np.any((cloth_points[:] == pt_list).all(axis=1)):
                idxs = np.where((cloth_points[:] == pt_list).all(axis=1))
                for id in idxs:
                    idx_grasped_list.append(id)
        # Array of x,y,z grasping points
        grasp_array = np.array(grasp_points)
        # Index of grasping points
        idx_grasped = np.array(idx_grasped_list).flatten()
        # Double check grasp array and cloth points grasped are the same
        if len(grasp_points) > 0:
            cloth_points_grasped = cloth_points[idx_grasped]
            np.testing.assert_allclose(grasp_array, cloth_points_grasped)
        return idx_grasped

    def _get_state_action(self, a_x, a_y, a_z):
        # Identify which are the grasped points
        idx_grasped = self.grasped_pts
        # Get the observation state
        obs_state = self.obs_1d
        # Create the actions array
        act_array = np.zeros(obs_state.shape, dtype=obs_state.dtype)
        gripped_array = np.zeros((obs_state.shape[0], 1), dtype=obs_state.dtype)
        if idx_grasped.size != 0:
            act_array[idx_grasped] = np.array([a_x, a_y, a_z])
        obs_state_action = np.hstack((obs_state, act_array, gripped_array))

        return obs_state_action

    def get_blender_rep(self, use_depth):
        """Ryan: put get_blender_rep in its own method so we can easily get RGBD images."""
        bhead = '/tmp/blender'
        if not os.path.exists(bhead):
            os.makedirs(bhead)

        # Step 1: make obj file using trimesh, and save to directory.
        wh = self.num_w
        #wh = self.num_h
        assert self.num_w == self.num_h  # TODO for now
        cloth = np.array([[p.x, p.y, p.z] for p in self.cloth.pts])
        assert cloth.shape[1] == 3, cloth.shape
        faces = []
        for r in range(wh-1):
            for c in range(wh-1):
                pp = r*wh + c
                faces.append( [pp,   pp+wh, pp+1] )
                faces.append( [pp+1, pp+wh, pp+wh+1] )
        tm = trimesh.Trimesh(vertices=cloth, faces=faces)
        # Handle file naming. Hopefully won't be duplicates!
        rank = '{}'.format(self._logger_idx)
        step = '{}'.format(str(self.num_steps).zfill(3))
        date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        base = 'gym-cloth-r{}-s{}-{}'.format(rank, step, date)
        tm_path = join(bhead, base)
        randnum = np.random.randint(1000000)  # np.random instead of np_random :-)
        tm_path = '{}_r{}.obj'.format(tm_path, str(randnum).zfill(7))
        tm.export(tm_path)

        # Step 2: call blender to get image representation.  We assume the
        # `blender` sub-package is at the same level as `envs`.  Use
        # __dirname__ to get path, then switch to `blender` dir.  Also deal
        # with data paths (for background frame) and different machines.

        init_side = 1 if self.cloth.init_side else -1
        #bfile = join(os.path.dirname(__file__), '../blender/get_image_rep.py') # 2.80
        bfile = join(os.path.dirname(__file__), '../blender/get_image_rep_279.py')
        frame_path = pkg_resources.resource_filename('gym_cloth', 'blender/frame0.obj')
        floor_path = pkg_resources.resource_filename('gym_cloth', 'blender/floor.obj')

        #Adi: Adding argument/flag for the oracle_reveal demonstrator
        #Adi: Adding argument/flag for using depth images
        #Adi: Adding argument for the floor obj path for more accurate depth images
        #Adi: Adding flag for domain randomization
        #Ryan: Adding hacky flags for fixed dom rand params per episode
        if sys.platform == 'darwin':
            subprocess.call([
                '/Applications/Blender/blender.app/Contents/MacOS/blender',
                '--background', '--python', bfile, '--', tm_path,
                str(self._hd), str(self._wd), str(init_side), self._init_type,
                frame_path, self._oracle_reveal, use_depth, floor_path,
                self.__add_dom_rand,
                ",".join([str(i) for i in self.dom_rand_params['c']]),
                ",".join([str(i) for i in self.dom_rand_params['n1']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_pos']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_deg']]),
                str(self.dom_rand_params['specular_max'])
                ]
            )
        else:
            subprocess.run([
                'blender', '--background', '--python', bfile, '--', tm_path,
                str(self._hd), str(self._wd), str(init_side), self._init_type,
                frame_path, self._oracle_reveal, use_depth, floor_path,
                self.__add_dom_rand,
                ",".join([str(i) for i in self.dom_rand_params['c']]),
                ",".join([str(i) for i in self.dom_rand_params['n1']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_pos']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_deg']]),
                str(self.dom_rand_params['specular_max'])
            ], stdout=subprocess.DEVNULL)
            # subprocess.call([
            #     'blender', '--background', '--python', bfile, '--', tm_path,
            #     str(self._hd), str(self._wd), str(init_side), self._init_type,
            #     frame_path, self._oracle_reveal, use_depth, floor_path,
            #     self.__add_dom_rand,
            #     ",".join([str(i) for i in self.dom_rand_params['c']]),
            #     ",".join([str(i) for i in self.dom_rand_params['n1']]),
            #     ",".join([str(i) for i in self.dom_rand_params['camera_pos']]),
            #     ",".join([str(i) for i in self.dom_rand_params['camera_deg']]),
            #     str(self.dom_rand_params['specular_max'])
            #     ]
            # )
        time.sleep(1)  # Wait a bit just in case

        # Step 3: load image from directory saved by blender.
        #Adi: Loading the occlusion state as well and saving it
        blender_path = tm_path.replace('.obj','.png')
        occlusion_path_pkl = tm_path.replace('.obj', '')
        with open(occlusion_path_pkl, 'rb') as fp:
            itemlist = pickle.load(fp)
            self._occlusion_vec = itemlist

        img = cv2.imread(blender_path)
        assert img.shape == (self._hd, self._wd, 3), \
                'error, shape {}, idx {}'.format(img.shape, self._logger_idx)

        if use_depth == 'True':
            # Smooth the edges b/c of some triangles.
            img = cv2.bilateralFilter(img, 7, 50, 50)
            if self._add_dom_rand:
                gval = self.dom_rand_params['gval_depth']
                img = np.uint8( np.maximum(0, np.double(img)-gval) )
            else:
                img = np.uint8( np.maximum(0, np.double(img)-50) )
        else:
            # Might as well randomize brightness if we're doing RGB.
            if self._add_dom_rand:
                gval = self.dom_rand_params['gval_rgb']
                img = self._adjust_gamma(img, gamma=gval)

        if self._add_dom_rand:
            # Apply some noise ONLY AT THE END OF EVERYTHING. I think it's
            noise = self.dom_rand_params['noise']
            img = np.minimum( np.maximum(np.double(img)+noise, 0), 255 )
            img = np.uint8(img)

        # If desired, save the images. Also save the resized version  --
        # make sure it's not done before we do noise addition, etc!
        #cv2.imwrite(blender_path, img)
        #img_small = cv2.resize(img, dsize=(100,100))  # dsize=(width,height)
        #cv2.imwrite(tm_path.replace('.obj','_small.png'), img_small)
        # careful! only if we want to override paths!!
        #cv2.imwrite(blender_path, img_small)
        #time.sleep(2)

        # Step 4: remaining book-keeping.
        if os.path.isfile(tm_path):
            os.remove(tm_path)
        return img

    def seed(self, seed=None):
        """Apply the env seed.

        See, for example:
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        We follow a similar convention by using an `np_random` object.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.logger.debug("Just re-seeded env to: {}".format(seed))
        return [seed]

    def save_state(self, cloth_file):
        """Save cloth.pts as a .pkl file.

        Be sure to supply a full path. Otherwise it saves under the `build/`
        directory somewhere.
        """
        with open(cloth_file, 'wb') as fh:
            pickle.dump({"pts": self.cloth.pts, "springs": self.cloth.springs}, fh)

    def step(self, action, initialize=False, cloth_updates=5):
        """Execute one action.
        Currently, actions are (pull fraction length).
        The top right corner is grasped and pulled using pull fraction length

        Originally, actions are parameterized as (grasp_point, pull fraction
        length, pull direction).
        """
        info = {}
        logger = self.logger
        exit_early = False
        # astr = self._act2str(action)

        # Truncate actions according to our bounds, then grip.
        low  = self.action_space.low
        high = self.action_space.high

        delta_x, delta_y, delta_z = action
        delta_x = max(min(delta_x, high[0]), low[0])
        delta_y = max(min(delta_y, high[1]), low[1])
        delta_z = max(min(delta_z, high[2]), low[2])

        # After this, we assume ranges {[0,1], [0,1],  [0,1], [-pi,pi]}.
        # Or if delta actions,         {[0,1], [0,1], [-1,1],   [-1,1]}.
        # Actually for non deltas, we have slack applied ...
        corner = 'top_right'
        self.gripper.grab_corner(corner)


        logger.info("         ======== EXECUTING ACTION: {} ========")
        logger.debug("Gripped at ({})".format(corner))
        logger.debug("Grabbed points: {}".format(self.gripper.grabbed_pts))
        logger.debug("Total grabbed: {}".format(len(self.gripper.grabbed_pts)))
        logger.debug("Action magnitudes: {:.4f}, {:.4f}, {:.4f}".format(delta_x, delta_y, delta_z))

        i = 0

        cloth_3d_obs = []
        initial_points = self._get_state_action(0., 0., 0.)
        # Perform the actual pulling
        self.gripper.adjust(x=delta_x, y=delta_y, z=delta_z)
        # Get the index of graspped points
        idx_pulled = self.grasped_pts
        if idx_pulled.size != 0:  # Add the actions to the cloth state observation
            initial_points[idx_pulled, 3:6] = np.array([delta_x, delta_y, delta_z])
            initial_points[idx_pulled, -1] = 1  # Add index for graspped point
        cloth_3d_obs.append(initial_points)
        # Update the cloth after pulling and get the state of the cloth
        for i in range(cloth_updates):
            self.cloth.update()
            updated_points = self._get_state_action(0., 0., 0.)
            updated_points[idx_pulled, -1] = 1
            if cloth_updates != 10:
                cloth_3d_obs.append(updated_points)

        if cloth_updates == 10:
            updated_points = self._get_state_action(0., 0., 0.)
            updated_points[idx_pulled, -1] = 1
            cloth_3d_obs.append(updated_points)
        # Test if the time-sequence has been well captured
        if initialize:
            return cloth_3d_obs
        self.num_steps += 1
        rew  = self._reward(action, exit_early)
        term = self._terminal()
        self.logger.info("Reward: {:.4f}. Terminal: {}".format(rew, term))
        self.logger.info("Steps/SimSteps: {}, {}".format(self.num_steps, self.num_sim_steps))
        info = {
            'num_steps': self.num_steps,
            'num_sim_steps': self.num_sim_steps,
            'out_of_bounds': self._out_of_bounds(),
            'obs_1d': cloth_3d_obs
        }
        return self.state, rew, term, info

    def _reward(self, action, exit_early):
        return self._fold_reward()

    def _fold_reward(self):
        "Check if the fold has succeeded"
        cloth_points = self.obs_1d
        # Get the top corners and bottom corners of the cloth
        # 0, 24, 600, 624
        top_left = cloth_points[0]
        top_right = cloth_points[24]
        bot_left = cloth_points[600]
        bot_right = cloth_points[624]

        # Compute the distance between each of the points
        dist_left = np.linalg.norm(top_left - bot_left)
        dist_right = np.linalg.norm(top_right - bot_right)
        self.logger.info("Distance left ={:.3f}".format(dist_left))
        self.logger.info("Distance right ={:.3f}".format(dist_right))

        # Compute the reward / sum of distance
        total_dist = dist_left + dist_right

        return total_dist

    def _terminal(self):
        "Check if the fold has succeeded"
        fold_dist = self._fold_reward()

        self.logger.info("Total distance ={:.3f}, threshold is ={:.3f}".format(fold_dist,
                                                                           _REWARD_THRESHOLDS[self.reward_type]))

        done = False
        if fold_dist < _REWARD_THRESHOLDS[self.reward_type]:  # Reward type is the foldding threshold
            done = True
        return done

    def reset(self):
        """Must call each time we start a new episode.

        Initializes to a new state, depending on the init 'type' in the config.

        `self.num_steps`: number of actions or timesteps in an episode.
        `self.num_sim_steps`: number of times we call `cloth.update()`.

        The above don't count any of the 'initialization' actions or steps --
        only those in the actual episode.

        Parameters
        ----------
        state: {"pts": [list of Points], "springs": [list of Springs]}
            If specified, load the cloth with this specific state and skip initialization.
        """
        reset_start = time.time()
        logger = self.logger
        cfg = self.cfg
        if self._start_state:
            self.cloth = cloth = Cloth(params=self.cfg,
                                       render=self.render_gl,
                                       random_state=self.np_random,
                                       render_port=self.render_port,
                                       state=copy.deepcopy(self._start_state))
        else:
            self.cloth = cloth = Cloth(params=self.cfg,
                                       render=self.render_gl,
                                       random_state=self.np_random,
                                       render_port=self.render_port)
        assert len(cloth.pts) == self.num_points, \
                "{} vs {}".format(len(cloth.pts), self.num_points)
        assert cloth.bounds[0] == self.bounds[0]
        assert cloth.bounds[1] == self.bounds[1]
        assert cloth.bounds[2] == self.bounds[2]
        self.gripper = gripper = Gripper(cloth, self.grip_radius,
                self.cfg['cloth']['height'], self.cfg['cloth']['thickness'])
        self.num_steps = 0
        self.num_sim_steps = 0
        self.have_tear = False

        if self.debug_viz:
            self.logger.info("Note: we set our config to visualize the init. We"
                    " will now play a video ...")
            nrows, ncols = 1, 2
            self.plt = plt
            self.debug_fig = plt.figure(figsize=(12*ncols,12*nrows))
            self.debug_ax1 = self.debug_fig.add_subplot(1, 2, 1)
            self.debug_ax2 = self.debug_fig.add_subplot(1, 2, 2, projection='3d')
            self.debug_ax2.view_init(elev=5., azim=-50.)
            self.plt.ion()
            self.plt.tight_layout()

        # Handle starting states, assuming we don't already have one.
        if not self._start_state:
            cloth_3d_obs = self._reset_actions()

        reset_time = (time.time() - reset_start) / 60.0
        logger.debug("Done with initial state, {:.2f} minutes".format(reset_time))

        # We shouldn't need to wrap around np.array(...) as self.state does that.
        # Ryan: compute dom rand params once per episode
        self.dom_rand_params['gval_depth'] = self.np_random.uniform(low=40, high=50) # really pixels ...
        self.dom_rand_params['gval_rgb'] = self.np_random.uniform(low=0.7, high=1.3)
        lim = self.np_random.uniform(low=-15.0, high=15.0)
        self.dom_rand_params['noise'] = self.np_random.uniform(low=-lim, high=lim, size=(self._wd, self._hd, 3))
        self.dom_rand_params['c'] = np.random.uniform(low=0.4, high=0.6, size=(3,))
        self.dom_rand_params['n1'] = np.random.uniform(low=-0.35, high=0.35, size=(3,))
        self.dom_rand_params['camera_pos'] = np.random.normal(0., scale=0.04, size=(3,)) # check get_image_rep_279.py for 'scale'
        self.dom_rand_params['camera_deg'] = np.random.normal(0., scale=0.90, size=(3,))
        self.dom_rand_params['specular_max'] = np.random.uniform(low=0.0, high=0.0) # check get_image_rep_279.py for 'high'
        obs = self.state
        return obs, cloth_3d_obs

    def _reset_actions(self):
        """Helper for reset in case reset applies action.
        """
        logger = self.logger
        cfg = self.cfg
        init_side = 1 if self.cloth.init_side else -1

        # Create a list to store all the cloth transitions
        cloth_3d_obs = []
        # Get the initial state-action of the cloth
        cloth_3d_obs.append(self._get_state_action(0.0, 0.0, 0.0))

        if self._init_type == 'tier4':
            logger.debug("Flat cloth")
            pass
        else:
            raise ValueError(self._init_type)

        logger.debug("STARTING COVERAGE: {:.2f}".format(self._compute_coverage()))
        logger.debug("STARTING VARIANCE: {:.2f}".format(self._compute_variance()))

        return cloth_3d_obs

    def get_random_action(self, atype='over_xy_plane'):
        """Retrieves random action.

        One way is to use the usual sample method from gym action spaces. Since
        we set the cloth plane to be in the range (0,1) in the x and y
        directions by default, we will only sample points over that range. This
        may or may not be desirable; we will sometimes pick points that don't
        touch any part of the cloth, in which case we just do a 'NO-OP'.

        The other option would be to sample any point that touches the cloth, by
        randomly picking a point from the cloth mesh and then extracting its x
        and y. We thus always touch something, via the 'naive cylinder' method.
        Though right now we don't support the delta actions here, for some reason.
        """
        if atype == 'over_xy_plane':
            return self.action_space.sample()
        elif atype == 'touch_cloth':
            assert not self._delta_actions
            pt = self.cloth.pts[ self.np_random.randint(self.num_points) ]
            length = self.np_random.uniform(low=0, high=1)
            angle = self.np_random.uniform(low=-np.pi, high=np.pi)
            action = (pt.x, pt.y, length, angle)
            if self._clip_act_space:
                action = ((pt.x - 0.5) * 2,
                          (pt.y - 0.5) * 2,
                          (length - 0.5) * 2,
                          angle / np.pi)
            return action
        else:
            raise ValueError(atype)

    def _out_of_bounds(self):
        """Detect if we're out of bounds, e.g., to stop an action.

        Currently, bounds are [0,1]. We add some slack for x/y bounds to
        represent cloth that drapes off the edge of the bed.  We should not be
        able to grasp these points, however.
        """
        pts = self.cloth.allpts_arr
        ptsx = pts[:,0]
        ptsy = pts[:,1]
        ptsz = pts[:,2]
        cond1 = np.max(ptsx) >= self.cloth.bounds[0] + self._slack
        cond2 = np.min(ptsx) < - self._slack
        cond3 = np.max(ptsy) >= self.cloth.bounds[1] + self._slack
        cond4 = np.min(ptsy) < - self._slack
        cond5 = np.max(ptsz) >= self.cloth.bounds[2]
        cond6 = np.min(ptsz) < 0
        outb = (cond1 or cond2 or cond3 or cond4 or cond5 or cond6)
        if outb:
           self.logger.debug("np.max(ptsx): {:.4f},  cond {}".format(np.max(ptsx), cond1))
           self.logger.debug("np.min(ptsx): {:.4f},  cond {}".format(np.min(ptsx), cond2))
           self.logger.debug("np.max(ptsy): {:.4f},  cond {}".format(np.max(ptsy), cond3))
           self.logger.debug("np.min(ptsy): {:.4f},  cond {}".format(np.min(ptsy), cond4))
           self.logger.debug("np.max(ptsz): {:.4f},  cond {}".format(np.max(ptsz), cond5))
           self.logger.debug("np.min(ptsz): {:.4f},  cond {}".format(np.min(ptsz), cond6))
        return outb

    def render(self, filepath, mode='human', close=False):
        """Much subject to change.

        If mode != 'matplotlib', spawn a child process rendering the cloth.
        As a result, you only need to call this once rather than every time
        step. The process is terminated with terminal(), so you must call
        it again after each episode, before calling reset().

        You will have to pass in the renderer filepath to this program, as the
        package will be unable to find it. To get the filepath from, for example,
        gym-cloth/examples/[script_name].py, run

        >>> this_dir = os.path.dirname(os.path.realpath(__file__))
        >>> filepath = os.path.join(this_dir, "../render/build")
        """
        if mode == 'matplotlib':
            self._debug_viz_plots()
        elif self.render_gl and not self.render_proc:
            owd = os.getcwd()
            os.chdir(filepath)
            dev_null = open('/dev/null','w')
            self.render_proc = subprocess.Popen(["./clothsim"], stdout=dev_null, stderr=dev_null)
            os.chdir(owd)

    # --------------------------------------------------------------------------
    # Random helper methods, debugging, etc.
    # --------------------------------------------------------------------------

    def _compute_variance(self):
        """Might want to use this instead of the internal method in reward()?
        """
        allpts = self.cloth.allpts_arr
        z_vals = allpts[:,2]
        variance = np.var(z_vals)
        if variance < 0.000001: # handle asymptotic behavior
            return 1000
        else:
            return 0.001 / variance

    def _compute_coverage(self):
        """Might want to use this instead of the internal method in _reward()?
        """
        points = np.array([[min(max(p.x,0),1), min(max(p.y,0),1)] for p in self.cloth.pts])
        try:
            # In 2D, this actually returns *AREA* (hull.area returns perimeter)
            hull = ConvexHull(points)
            coverage = hull.volume
        except scipy.spatial.qhull.QhullError as e:
            logging.exception(e)
            #_save_bad_hull(points)
            coverage = 0
        return coverage

    def _debug_viz_plots(self):
        """Use `plt.ion()` for interactive plots, requires `plt.pause(...)` later.

        This is for the debugging part of the initialization process. It's not
        currently meant for the actual rendering via `env.render()`.
        """
        plt = self.plt
        ax1 = self.debug_ax1
        ax2 = self.debug_ax2
        eps = 0.05

        ax1.cla()
        ax2.cla()
        pts  = self.cloth.noncolorpts_arr
        cpts = self.cloth.colorpts_arr
        ppts = self.cloth.pinnedpts_arr
        if len(pts) > 0:
            ax1.scatter(pts[:,0], pts[:,1], c='g')
            ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c='g')
        if len(cpts) > 0:
            ax1.scatter(cpts[:,0], cpts[:,1], c='b')
            ax2.scatter(cpts[:,0], cpts[:,1], cpts[:,2], c='b')
        if len(ppts) > 0:
            ax1.scatter(ppts[:,0], ppts[:,1], c='darkred')
            ax2.scatter(ppts[:,0], ppts[:,1], ppts[:,2], c='darkred')
        ax1.set_xlim([0-eps, 1+eps])
        ax1.set_ylim([0-eps, 1+eps])
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_zlim([0, 1])
        plt.pause(0.0001)

    def _save_matplotlib_img(self, target_dir=None):
        """Save matplotlib image into a target directory.
        """
        #Adi: Quick test
        print("SAVING AN IMAGE!!!")
        plt = self.plt
        if target_dir is None:
            target_dir = (self.fname_log).replace('.log','.png')
        print("Note: saving matplotlib img of env at {}".format(target_dir))
        plt.savefig(target_dir)

    def _setup_logger(self):
        """Set up the logger (and also save the config w/similar name).

        If you create a new instance of the environment class in the same
        program, you will get duplicate logging messages. We should figure out a
        way to fix that in case we want to scale up to multiple environments.

        Daniel TODO: this is going to refer to the root logger and multiple
        instantiations of the environment class will result in duplicate
        logging messages to stdout. It's harmless wrt environment stepping.
        """
        cfg = self.cfg
        dstr = '-{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        filename = (cfg['log']['file']).replace('.log',dstr)
        if self._logger_idx is not None:
            filename = filename.replace('.log',
                    '_rank_{}.log'.format(str(self._logger_idx).zfill(2)))
        logging.basicConfig(
                level=_logging_setup_table[cfg['log']['level']],
                filename=filename,
                filemode='w')

        # Define a Handler which writes messages to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(_logging_setup_table[cfg['log']['level']])

        # Set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s '
                                      '%(message)s', datefmt='%m-%d %H:%M:%S')

        # Tell the handler to use this format, and add handler to root logger
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        if self._logger_idx is not None:
            self.logger = logging.getLogger("cloth_env_{}".format(self._logger_idx))
        else:
            self.logger = logging.getLogger("cloth_env")

        # Finally, save config file so we can exactly reproduce parameters.
        json_str = filename.replace('.log','.json')
        with open(json_str, 'w') as fh:
            json.dump(cfg, fh, indent=4, sort_keys=True)
        self.fname_log = filename
        self.fname_json = json_str

    def _act2str(self, action):
        """Turn an action into something more human-readable.
        """
        if self._delta_actions:
            x, y, dx, dy = action
            astr = "({:.2f}, {:.2f}), deltax {:.2f}, deltay {:.2f}".format(
                    x, y, float(dx), float(dy))
        else:
            x, y, length, direction = action
            if self._clip_act_space:
                astr = "({:.2f}, {:.2f}), length {:.2f}, angle {:.2f}".format(
                    x, y, float(length), float(direction))
                astr += "  Re-scaled: ({:.2f}, {:.2f}), {:.2f}, {:.2f}".format(
                    (x/2)+0.5, (y/2)+0.5, (length/2)+0.5, direction*np.pi)
            else:
                astr = "({:.2f}, {:.2f}), length {:.2f}, angle {:.2f}".format(
                    x, y, float(length), float(direction))
        return astr

    def _convert_action_to_clip_space(self, a):
        # Help out with all the clipping and stuff.
        if not self._clip_act_space:
            return a
        if self._delta_actions:
            newa = ((a[0]-0.5)*2, (a[1]-0.5)*2,         a[2],       a[3])
        else:
            newa = ((a[0]-0.5)*2, (a[1]-0.5)*2, (a[2]-0.5)*2, a[3]/np.pi)
        return newa

    def _adjust_gamma(self, image, gamma=1.0):
        """For darkening images.

        Builds a lookup table mapping the pixel values [0, 255] to their
        adjusted gamma values.
        https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 \
                for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
