### This project is a very base example of how to implement PX4 / MAVSDK / SB3 training
- The goal is to deploy this onto a quadcopter to have a full training pipeline from simulation, simulation in the loop (SITL), hardware in the loop (HITL) to actual deployment
- Since PX4 is asynchronous, the goal here is to create a synchronous wrapper around the PX4-Gazebo environment to expose SB3 to the environment to allow for agent learning
- Development is on a HolyBro drone with 8 rotors, 2 electronic speed controllers (ESCs), and 1 flight controller
- > The drone also has 1 NVIDIA Jetson Orin Nano 8GB with option for vision control (which will be done after hovering from RL is exhibited)
