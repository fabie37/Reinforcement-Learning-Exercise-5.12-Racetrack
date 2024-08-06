# Reinforcement Learning: Exercise 5.12 Racetrack

This is a single solution to Sutton and Barto's Reinforcement Learning Exercise. Racetrack. 

This uses pygame to provide a visual gui to the agent learning to turn a corner in a racetrack. 

Installing
- Once you have cloned this repo create a virtual enviroment

```
$: python -m venv ./venv
$: source ./venv/Scripts/activate
    ^ check for your operating system: different on Mac, Linux and Windows
```
- Then install dependances 
```
$: python -m pip install -r requirements.txt
```

- Then run code
```
$: python race_track.py
```

Program States:
- Building: Use GUI to build your racetrack
- Playing: Use your arrow keys to manually move the agent
- Learning: Agent will run the RL Algorithm (Use The Top button to switch from learning to picking the optimal action)

RL Algorithm:
- Algorithm: Monte Carlo On Policy using Every Visit
- The exercise doesn't specify which one to use. 
- Feel free to implement off policy and branch this repo

GUI:
- Use this to draw your track
- Left Mouse Button (Can be held down): Change square to next state
- Right Mouse Button (Can be held down): Change square to prev state

The agent: 
- Max Velocity: 5
- Actions 9: 
  - Increase/Decrease Velocity in each row/col dimension

The Environment:
- Grid Squares
- States
  - Boundary: Agent Dies in this state [Black]
  - Open: Agent is free to move [Grey/White]
  - Start: Agent spawns on this square [Red]
  - Finish: Agent Wins an Episode [Green]
