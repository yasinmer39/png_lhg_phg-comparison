import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def PID(error, coef_int, prev, kp, ki ,kd):
    coef_int += error  # Integral term
    out = error * kp + coef_int * ki + (error - prev) * kd  # PID output
    prev = error  # Update previous error

    return out, coef_int, prev  # Return updated integral and previous values

def state_space_model(x, u):
    """
    Nonlinear state-space model.
    x: [x, y, theta, v (vx,vy)]
    u: [u1 (acceleration), u2 (angular velocity)]
    Returns: dx/dt
    """
    x_pos, y_pos, theta, v = x
    u1, u2 = u
    dx = np.zeros_like(x)
    dx[0] = v * np.cos(theta)
    dx[1] = v * np.sin(theta)
    dx[2] = u2
    dx[3] = u1
    return dx

def calculate_los_angle(pursuer, target):
    """
    Calculate the Line of Sight angle between pursuer and target
    Returns angle in degrees (0-360) measured clockwise from true north
    """
    delta_x = target["iha_enlem"] - pursuer["iha_enlem"]
    delta_y = target["iha_boylam"] - pursuer["iha_boylam"]
    
    # Calculate LOS angle in radians and convert to degrees
    los_angle = np.rad2deg(np.arctan2(delta_y, delta_x))
    
    # Convert to 0-360 range
    los_angle = los_angle % 360
    
    return los_angle

def follow_target(pursuer, target, controller, gain=1.0):
    """
    Adjust the pursuer's heading to follow the target using LOS guidance.
    gain: Proportional gain for adjusting the heading.
    """
    los_angle = calculate_los_angle(pursuer, target)
    current_heading = pursuer["iha_yonelme"]
    
    # Calculate heading difference
    heading_diff = (los_angle - current_heading) % 360
    if heading_diff > 180:
        heading_diff -= 360
    
    # Adjust roll to reduce heading difference
    controller.set_roll(current_heading + gain * heading_diff)

class KalmanFilter:


    def __init__(self, initial_x, initial_y, initial_vx, initial_vy):

        dt = 1/2

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.statePre = np.array([[initial_x], [initial_y], [initial_vx], [initial_vy]], np.float32) * 0.03


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = float(predicted[0]), float(predicted[1])
        return x, y
    
class MissileController:
    def __init__(self):
        self.max_roll_rate = 20  # Maximum degrees per update
        self.base_speed = 0.1    # Base speed units per update
        self.max_speed = 0.3     # Maximum speed units per update
        
        # Current state
        self.current_speed = 0.0
        self.target_roll = 0.0
        self.target_thrust = 0.0
        
    def set_roll(self, angle):
        """Set desired roll angle in degrees"""
        self.target_roll = angle % 360
        
    def set_thrust(self, thrust):
        """Set desired thrust (0.0 to 1.0)"""
        self.target_thrust = max(0.0, min(1.0, thrust))
        
    def update(self, payload):
        # Update heading based on roll command
        current_heading = payload["iha_yonelme"]
        heading_diff = (self.target_roll - current_heading) % 360
        if heading_diff > 180:
            heading_diff -= 360
            
        # Apply maximum roll rate limit
        roll_change = max(-self.max_roll_rate, min(self.max_roll_rate, heading_diff))
        payload["iha_yonelme"] = (current_heading + roll_change) % 360
        
        # Update speed based on thrust command
        self.current_speed = self.base_speed + (self.max_speed - self.base_speed) * self.target_thrust
        
        # Calculate movement based on current heading and speed
        rad = np.deg2rad(payload["iha_yonelme"])
        payload["iha_enlem"] += self.current_speed * np.cos(rad)
        payload["iha_boylam"] += self.current_speed * np.sin(rad)
        
        return payload

def plot_arrow(pos, color="red"):
    rad = np.deg2rad(pos["iha_yonelme"])
    arrow_end_x = pos["iha_enlem"] + 0.1 * np.cos(rad)
    arrow_end_y = pos["iha_boylam"] + 0.1 * np.sin(rad)
    plt.arrow(pos["iha_enlem"], pos["iha_boylam"], arrow_end_x - pos["iha_enlem"], arrow_end_y - pos["iha_boylam"],
              head_width=0.03, head_length=0.05, fc=color, ec=color)


def plot_text(your_pos, pos):
    if your_pos["iha_irtifa"] == pos["iha_irtifa"]:
        p = ""
    elif your_pos["iha_irtifa"] < pos["iha_irtifa"]:
        p = " +"
    else:
        p = " -"
    plt.text(pos["iha_enlem"], pos["iha_boylam"], str(pos["takim_numarasi"]) + p)


def plot_point(pos_array):
    for pos in pos_array:
        plt.plot(pos[0], pos[1], 'ro')


def plot_line(pos_array, linestyle='solid'):
    pos_array = np.array(pos_array)
    if linestyle=="dashed":
        color="blue"
    else:
        color=None
    plt.plot(pos_array[:, 0], pos_array[:, 1], linestyle=linestyle, color=color)

class generate_map:

    def __init__(self, id, lat_range, lon_range, points):
        self.id = id
        self.std_dev = 1
        self.lat_range = lat_range
        self.lon_range = lon_range
        y = np.linspace(lat_range[0], lat_range[1], int((lat_range[1] - lat_range[0]) / 0.01))
        x = np.linspace(lon_range[0], lon_range[1], int((lon_range[1] - lon_range[0]) / 0.01))

        self.lat, self.lon = np.meshgrid(y, x)
        self.contour = np.zeros(self.lat.shape)
        self.cache = dict()

        for (x, y) in points:
            self.contour += np.exp(-((self.lat - x) ** 2 + (self.lon - y) ** 2) / (2 * self.std_dev))

        self.contour = np.clip(self.contour, -1, 1)

    def set_border(self, contour):
        n = 30
        for i in range(n):
            contour[i, i:-i - 1] = 1 / (i + 1)
            contour[-(1 + i), i:-i] = 1 / (i + 1)
            contour[i:-i - 1, i] = 1 / (i + 1)
            contour[i:-i - 1, -(1 + i)] = 1 / (i + 1)
        contour[-1] = 1

    def update(self, telem):
        contour = np.zeros(self.lat.shape)  # Start with a clean slate each time

        for pos in telem["konumBilgileri"]:
            self.history(pos)
            if pos["takim_numarasi"] != self.id:
                # Blue background for payload2 (negative values)
                contour -= np.exp(-((self.lat - pos["iha_enlem"]) ** 2 + (self.lon - pos["iha_boylam"]) ** 2) / (2 * self.std_dev))
            else:
                # Red background for payload1 (positive values)
                contour += np.exp(-((self.lat - pos["iha_enlem"]) ** 2 + (self.lon - pos["iha_boylam"]) ** 2) / (2 * self.std_dev))
                your_pos = pos
                
        self.set_border(contour)
        return contour, your_pos

    def history(self, pos):
        if pos["takim_numarasi"] not in self.cache:
            self.cache[pos["takim_numarasi"]] = deque([[pos["iha_enlem"], pos["iha_boylam"]]], maxlen=10)
        else:
            self.cache[pos["takim_numarasi"]].append([pos["iha_enlem"], pos["iha_boylam"]])

def is_within_trim_distance(pos1, pos2, trim):
    return abs(pos2["iha_enlem"] - pos1["iha_enlem"]) < trim and abs(pos2["iha_boylam"] - pos1["iha_boylam"]) < trim


def is_within_test_range(pos, lat_range, lon_range):
    return lat_range[0] < pos["iha_enlem"] < lat_range[1] and lon_range[0] < pos["iha_boylam"] < lon_range[1]


def should_plot(pos, your_pos, env, trim):
    within_trim_distance = trim != 0 and is_within_trim_distance(pos, your_pos, trim)
    within_test_range = trim == 0 and is_within_test_range(pos, env.lat_range, env.lon_range)
    return within_test_range or within_trim_distance

def circular_movement(center_x, center_y, radius, angle):
    """Calculate position on a circle based on angle"""
    x = center_x + radius * np.cos(np.deg2rad(angle))
    y = center_y + radius * np.sin(np.deg2rad(angle))
    return x, y

dt = 0.1  # Time step for state space model

# Initialize state vectors [x, y, theta, v] for both payloads
state1 = np.array([41.0, 26.0, 0.0, 0.0])  # Initial state for payload1
state2 = np.array([42.0, 27.0, 0.0, 0.0])  # Initial state for payload2

# Update payload positions based on states
payload1 = {
    "takim_numarasi": 1,
    "iha_enlem": state1[0],
    "iha_boylam": state1[1],
    "iha_irtifa": 25,
    "iha_dikilme": 0,
    "iha_yonelme": np.rad2deg(state1[2]),  # Convert theta to degrees
    "iha_yatis": 0,
    "zaman_farki": 93
}

payload2 = {
    "takim_numarasi": 2,
    "iha_enlem": state2[0],
    "iha_boylam": state2[1],
    "iha_irtifa": 25,
    "iha_dikilme": 0,
    "iha_yonelme": np.rad2deg(state2[2]),
    "iha_yatis": 0,
    "zaman_farki": 74
}

test = generate_map(1, [39, 44], [20, 30], [(40, 27)])
trim = 0

# Initialize angles for circular movement
angle1 = 0
angle2 = 180  # Start opposite to payload1

controller1 = MissileController()
controller2 = MissileController()

# Adjusted PID controller parameters
kp = 2.0  # Increase proportional gain for faster response
ki = 0.1  # Increase integral gain to reduce steady-state error
kd = 0.9  # Increase derivative gain to dampen oscillations

# Initialize integral and previous error for PID
integral = 0.0
previous_error = 0.0

for i in range(1000):
    los_angle = calculate_los_angle(payload1, payload2)
    current_heading = payload1["iha_yonelme"]
    
    # Calculate heading difference
    heading_diff = (los_angle - current_heading) % 360
    if heading_diff > 180:
        heading_diff -= 360
    
    # Use PID controller to calculate turning rate
    turning_rate, integral, previous_error = PID(heading_diff, integral, previous_error, kp, ki, kd)
    
    # Control inputs for payload1
    u1 = np.array([0.5,  # Constant acceleration
                   np.deg2rad(turning_rate)])  # Turning rate from PID
    
    # Control inputs for payload2 (target) - example circular motion
    u2 = np.array([0.2,  # Constant acceleration
                   0.2])  # Constant turning rate
    
    # Update states using state space model
    dx1 = state_space_model(state1, u1)
    dx2 = state_space_model(state2, u2)
    
    state1 = state1 + dx1 * dt
    state2 = state2 + dx2 * dt
    
    # Update payloads with new states
    payload1.update({
        "iha_enlem": state1[0],
        "iha_boylam": state1[1],
        "iha_yonelme": np.rad2deg(state1[2]) % 360
    })
    
    payload2.update({
        "iha_enlem": state2[0],
        "iha_boylam": state2[1],
        "iha_yonelme": np.rad2deg(state2[2]) % 360
    })
    
    telem = {"sunucuSaati": {
        "saat": 6,
        "dakika": 53,
        "saniye": 42,
        "milisaniye": 500
    }, "konumBilgileri": [payload1, payload2]}

    contour, your_pos = test.update(telem)

    plt.clf()
    cp = plt.contourf(test.lat, test.lon, contour, cmap='coolwarm', levels=25)
    plt.colorbar(cp)
    for pos in telem["konumBilgileri"]:
        if should_plot(pos, your_pos, test, trim):
            plot_text(your_pos, pos)
            plot_line(test.cache[pos["takim_numarasi"]])
            plot_arrow(pos)

    plt.pause(0.5)