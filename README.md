# 🚀 **Satellite Collision Detection and Avoidance using Machine Learning & RL**  

This project simulates satellite and space debris data, detects potential collisions, visualizes orbits, and applies machine learning & reinforcement learning to predict and optimize satellite maneuvers.  

---

## 📌 **Installation**  

Before running the code, install the required dependencies:  

```bash
pip install numpy pandas matplotlib scikit-learn scipy stable-baselines3 gym gymnasium shimmy
```

---

## 🌍 **Synthetic Dataset Generation**  

The project generates synthetic satellite and debris data in a 3D space:  

- **Satellites**: Defined by their positions `(x, y, z)` and velocities `(vx, vy, vz)`.  
- **Debris**: Randomly distributed space debris with varying positions and velocities.  

```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define satellites
satellites = np.array([
    [1, 7000, 0, 0, 0, 7.5, 0],  
    [2, 0, 7000, 0, -7.5, 0, 0]
])

# Generate random debris positions and velocities
debris = np.random.uniform(low=-10000, high=10000, size=(100, 6))
debris = np.hstack((np.arange(1, 101).reshape(-1, 1), debris))  # Add IDs

# Convert to Pandas DataFrame
satellites_df = pd.DataFrame(satellites, columns=["id", "x", "y", "z", "vx", "vy", "vz"])
debris_df = pd.DataFrame(debris, columns=["id", "x", "y", "z", "vx", "vy", "vz"])

print(satellites_df.head())
print(debris_df.head())
```

---

## ⚠️ **Collision Detection using Euclidean Distance**  

Using **SciPy's** `cdist()`, the script detects debris within **100 km** of a satellite.  

```python
from scipy.spatial.distance import cdist

def detect_collisions(satellites, debris, threshold=100):
    collisions = []
    for _, sat in satellites.iterrows():
        sat_pos = sat[["x", "y", "z"]].values.reshape(1, -1)
        debris_pos = debris[["x", "y", "z"]].values
        distances = cdist(sat_pos, debris_pos)
        close_debris = np.where(distances < threshold)[1]
        for debris_id in close_debris:
            collisions.append((sat["id"], debris.iloc[debris_id]["id"], distances[0, debris_id]))
    return collisions

# Detect and print collisions
collisions = detect_collisions(satellites_df, debris_df)
for col in collisions:
    print(f"Satellite {col[0]} and Debris {col[1]} are {col[2]:.2f} km apart")
```

---

## 🌌 **Orbit Visualization in 3D**  

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(satellites_df["x"], satellites_df["y"], satellites_df["z"], c="blue", label="Satellites")
ax.scatter(debris_df["x"], debris_df["y"], debris_df["z"], c="red", label="Debris", alpha=0.5)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.legend()
plt.title("Satellite and Debris Positions")
plt.show()
```

---

## 🧠 **Machine Learning for Collision Prediction**  

Using **Random Forest Classifier** to predict potential collisions based on relative distance and velocity.  

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Normalize Data
scaler = StandardScaler()
satellites_normalized = scaler.fit_transform(satellites_df[["x", "y", "z", "vx", "vy", "vz"]])
debris_normalized = scaler.transform(debris_df[["x", "y", "z", "vx", "vy", "vz"]])

# Feature Engineering
def create_features(satellites, debris):
    features = []
    for _, sat in satellites.iterrows():
        sat_pos = sat[["x", "y", "z"]].values
        sat_vel = sat[["vx", "vy", "vz"]].values
        for _, deb in debris.iterrows():
            deb_pos = deb[["x", "y", "z"]].values
            deb_vel = deb[["vx", "vy", "vz"]].values
            rel_distance = np.linalg.norm(sat_pos - deb_pos)
            rel_velocity = np.linalg.norm(sat_vel - deb_vel)
            features.append([rel_distance, rel_velocity])
    return np.array(features)

X = create_features(satellites_df, debris_df)
y = np.random.randint(0, 2, size=len(X))  # Simulated labels (Replace with real collision data)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate Model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
```

---

## 🤖 **Reinforcement Learning for Orbit Optimization**  

Using **Stable-Baselines3 PPO** for satellite trajectory optimization.  

```bash
pip install stable-baselines3 gymnasium
```

```python
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

class SatelliteEnv(gym.Env):
    def __init__(self):
        super(SatelliteEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))  # Thrust in x, y, z
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))  # Position and velocity

    def reset(self, seed=None, options=None):
        return np.zeros(6), {}

    def step(self, action):
        new_state = np.zeros(6)  # Placeholder for real dynamics
        reward = 0  # Reward function based on collision avoidance and fuel usage
        done = False
        return new_state, reward, done, False, {}

# Create environment
env = SatelliteEnv()

# Train PPO Model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("Training complete!")
```

---

## 📜 **Summary**  

| Feature | Description |
|---------|-------------|
| **Synthetic Data** | Generates random satellite and debris positions/velocities |
| **Collision Detection** | Uses Euclidean distance to detect potential collisions |
| **Visualization** | 3D plot of satellite and debris positions |
| **Machine Learning** | Predicts collision probability using Random Forest |
| **Reinforcement Learning** | Optimizes satellite trajectory using PPO |

🚀 **This project is a step towards real-world space debris tracking and autonomous collision avoidance!** 🌍💫  

---

## 📌 **Next Steps**  
- Integrate real satellite tracking data (e.g., TLE from NORAD).  
- Improve RL reward function for efficient fuel usage.  
- Deploy the model for real-time space monitoring.  

---

## 👨‍💻 **Author**  
**Your Name**  
🔗 [GitHub](https://github.com/your-profile) | 📨 [Email](mailto:your-email@example.com)  

---

🎯 **If you find this useful, give it a ⭐ on GitHub!** 🚀
