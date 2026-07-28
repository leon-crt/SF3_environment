import subprocess
import socket
import win32process
import win32gui
import win32api
import win32con
import numpy as np
import gymnasium as gym
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
emu_path = PACKAGE_DIR / "fbneo" / "env_emu.exe"

HOST = "127.0.0.1"
PORT = 42069

# TODO:
#       - use all needed keys for output tensordict

def enumWindowsProc(hwnd, lParam):
    if (lParam is None) or ((lParam is not None) and win32process.GetWindowThreadProcessId(hwnd)[1] == lParam):
        text = win32gui.GetWindowText(hwnd)
        if text:
            win32api.SendMessage(hwnd, win32con.WM_CLOSE)

class SF3Env(gym.Env):
    metadata = {"render_modes": ["human", "turbo"], "render_fps": 60, "play_modes": ["cpu", "free", "selfplay"]}

    def __init__(self, render_mode="human", mode="cpu", threshold=0.5, P1_ch='Ryu', P2_ch='Ken'):
        assert mode in self.metadata["play_modes"]
        self.mode = mode
        self.P1_ch = P1_ch
        self.P2_ch = P2_ch
        min_state_limits = np.array([93, -42, 0, 0, 0, 0, 0, 0])
        max_state_limits = np.array([928, 226, 161, 336, 70, 1, 1, 1])
        self.observation_space = gym.spaces.Dict({
            'player_state': gym.spaces.Box(min_state_limits, max_state_limits, (8,), np.float32),
            'opponent_state': gym.spaces.Box(min_state_limits, max_state_limits, (8,), np.float32),
            'opponent_inputs': gym.spaces.Box(0,1,(12,), np.int8),
        })
        self.action_space = gym.spaces.Dict({
            'player_inputs': gym.spaces.MultiBinary(12)
        })
        self._player_state = np.array([-100] * 8)
        self._opp_state = np.array([-100] * 8)
        self._opp_inputs = np.array([-1] * 12)
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.socket = None
        self.threshold = threshold
    
    def _get_obs(self):
        return {
            'player_state': self._player_state,
            'opponent_state': self._opp_state,
            'opponent_inputs': self._opp_inputs
        }
    
    def flatten_obs(self, obs):
        return  np.concat([obs["player_state"], obs['opponent_state'], obs['opponent_inputs']])
    
    def _parse_state(self, state):
        res = state.split(sep=',')[:-1]
        res = [int(x) for x in res]
        return res
    
    def _update_fields(self, state):
        self._player_state = np.array(state[:len(self._player_state)], dtype=np.float32)
        self._opp_state = np.array(state[len(self._player_state):-len(self._opp_inputs)], dtype=np.float32)
        self._opp_inputs = np.array(state[-len(self._opp_inputs):], dtype=np.int8)
    
    def reward(self, prev_obs, new_obs):
        r = 0
        health_pl = 2
        health_opp = 10
        meter_pl = 3
        stun_pl = 5
        stun_opp = 13
        got_hit = False

        # get arrays from the observation dictionaries
        prev_state = self.flatten_obs(prev_obs)
        new_state = self.flatten_obs(new_obs)

        # check health values
        if new_state[health_pl] == 0: # terminal failure (player lost)
            r = -10
        elif new_state[health_opp] == 0: # terminal success (opponent lost)
            r = 10
        if prev_state[health_pl] > new_state[health_pl]: # player got hit
            r -= 0.3
            got_hit = True
        if prev_state[health_opp] > new_state[health_opp]: # opponent got hit
            r += 0.3
        # time-step penalty
        if new_state[health_pl] <= new_state[health_opp]:
            r -= 0.05
        
        # stun related rewards and penalties
        if new_state[stun_pl] == 1 and prev_state[stun_pl] == 0:
            r -= 0.5
        if new_state[stun_opp] == 1 and prev_state[stun_opp] == 0:
            r += 0.5
        
        # meter related rewards and penalties
        if new_state[meter_pl] > prev_state[meter_pl] and (not got_hit):
            r += 0.1
        
        return r
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # check if emulator instance is already running and close it in case it is
        if self.window != None:
            win32gui.EnumWindows(enumWindowsProc, self.window.pid)
        # start emulator
        self.window = subprocess.Popen([str(emu_path), "sfiii3nr1", "./sf3_env.lua"])

        # establish tcp connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((HOST, PORT))
        try:
            # send configuration
            config = f"{self.mode};{self.render_mode}"
            self.socket.send(bytes(config + '\r\n', "utf-8"))

            # receive first state
            data = self.socket.recv(100)
            # catch graceful disconnection
            if not data:
                raise ConnectionError(f"Client {HOST} disconnected gracefully.")
            
            data = data.decode('utf-8')
        
            # decode the string and remove the padding
            data = data.replace('#', '')
            state = self._parse_state(data)
            # update class variables
            self._update_fields(state)
            
        except (ConnectionResetError, BrokenPipeError) as e:
            # catch abrupt network cut
            raise ConnectionError(f"Connection with {HOST} was interrupted abruptly: {e}")

        return self._get_obs(), {}

    # Step function for when mode=selfplay so that inputs for both players are provided
    def step(self, action_player, action_opp=None):
        action_player = list(action_player)
        # format and send the action based on whether we are doing self play or not
        if action_opp != None:
            data_p1 = str(action_player)[1:-1].replace(' ', '')
            data_p2 = str(action_opp)[1:-1].replace(' ', '')
            data = data_p1 + ';' + data_p2
        else:
            data = str(action_player)[1:-1].replace(' ', '')
        self.socket.send(bytes(data + '\r\n', "utf-8"))

        terminated = False

        # receive next game state
        data = self.socket.recv(100)

        # catch graceful disconnection
        if not data:
            raise ConnectionError(f"Client {HOST} disconnected gracefully.")
        
        data = data.decode('utf-8')
        data = data.replace('#', '')
        # terminal state
        if data[-1] == 'R':
            data = self._parse_state(data[:-1])
            terminated = True
        else:
            data = self._parse_state(data)
        
        previous_state = self._get_obs()
        # update class variables
        self._update_fields(data)
        observation = self._get_obs()

        # compute reward
        r = self.reward(previous_state, observation)

        return observation, r, terminated, False, {}

    def close(self):
        self.socket.close()
        if self.window != None:
            win32gui.EnumWindows(enumWindowsProc, self.window.pid)
        self.socket = None