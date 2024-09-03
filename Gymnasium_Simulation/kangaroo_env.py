import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class KangarooEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 250, # Gangaroo I changed it to 250,
    }

    def __init__(
        self,
        vertical_reward_weight=0.75,
        horizontal_reward_weight=2,
        forward_reward_weight=1,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-200.0, 200.0),
        healthy_z_range=(-1, float("inf")),
        healthy_angle_range=(-1, 0.4),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            vertical_reward_weight,
            horizontal_reward_weight,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._vertical_reward_weight = vertical_reward_weight
        self._horizontal_reward_weight = horizontal_reward_weight
        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
                # gangaroo
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
                # gangaroo
            )

        MujocoEnv.__init__(
            self,
            "kangaroo.xml", # gangaroo
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        #print(f"state:{healthy_state}      z:{healthy_z}       angle:{healthy_angle}")
        #print(f"state:{np.sum(state)}      z:{z}       angle:{angle}")
        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        #print(f"is_healthy{self.is_healthy}")
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False#terminate when if_healthy is false
        #print(f"is_healthy{self.is_healthy}")
        return terminated

    def _get_obs(self):
        # gangaroo
        # What about the other observations?
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        # gangaroo
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = - self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        vertical_reward = (- self.data.qpos[1]) * self._vertical_reward_weight
        horizontal_reward = (- self.data.qpos[0]) * self._horizontal_reward_weight

        rewards = forward_reward + healthy_reward + vertical_reward + horizontal_reward
        costs = ctrl_cost

        #print(f"cost:{ctrl_cost}        forward:{forward_reward}        health:{healthy_reward}     vertical:{vertical_reward}      hori:{horizontal_reward}")

        observation = self._get_obs()
        
        #print(f"{self._get_obs()}\n\n")
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        #print(f"return:{terminated}")
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
