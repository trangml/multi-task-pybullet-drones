


# multi-task-pybullet-drones

This repository builds off the work done in [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) with extensions focused on exploring multi-task learning. 

Key changes include a YAML configuration system for rapidly testing environment designs, rewards, and terminations. 
New environments can be found in the [env] (https://github.com/trangml/multi-task-pybullet-drones/gym_pybullet_drones/envs) directory.
These use configurations of the reward and terminations that can be found in [rewards] (https://github.com/trangml/multi-task-pybullet-drones/gym_pybullet_drones/envs/single_agent_rl/rewards)


[Simple](https://en.wikipedia.org/wiki/KISS_principle) OpenAI [Gym environment](https://gym.openai.com/envs/#classic_control) based on [PyBullet](https://github.com/bulletphysics/bullet3) for multi-agent reinforcement learning with quadrotors 

<img src="files/readme_images/helix.gif" alt="formation flight" width="350"> <img src="files/readme_images/helix.png" alt="control info" width="450">

- The default `DroneModel.CF2X` dynamics are based on [Bitcraze's Crazyflie 2.x nano-quadrotor](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf)





## Requirements and Installation
The repo was written using *Python 3.7* with [`conda`](https://github.com/JacopoPan/a-minimalist-guide#install-conda) on *macOS 10.15* and tested with *Python 3.8* on *macOS 12*, *Ubuntu 20.04*




### On *macOS* and *Ubuntu*
Major dependencies are [`gym`](https://gym.openai.com/docs/),  [`pybullet`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#), 
[`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html), and [`rllib`](https://docs.ray.io/en/master/rllib.html)

Video recording requires to have [`ffmpeg`](https://ffmpeg.org) installed, on *macOS*
```bash
$ brew install ffmpeg
```
On *Ubuntu*
```bash
$ sudo apt install ffmpeg
```

*macOS* with Apple Silicon (like the M1 Air) can only install grpc with a minimum Python version of 3.9 and these two environment variables set:
```bash
$ export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
$ export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
```

The repo is structured as a [Gym Environment](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
and can be installed with `pip install --editable`
```
$ conda create -n drones python=3.8 # or 3.9 on Apple Silicon, see the comment on grpc above
$ conda activate drones
$ pip3 install --upgrade pip
$ git clone https://github.com/utiasDSL/gym-pybullet-drones.git
$ cd gym-pybullet-drones/
$ pip3 install -e .
```
On Ubuntu and with a GPU available, optionally uncomment [line 203](https://github.com/utiasDSL/gym-pybullet-drones/blob/fab619b119e7deb6079a292a04be04d37249d08c/gym_pybullet_drones/envs/BaseAviary.py#L203) of `BaseAviary.py` to use the [`eglPlugin`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.778da594xyte)







## gym-pybullet-drones Citation
If you wish, please cite our work [(link)](https://arxiv.org/abs/2103.02142) as
```
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={},
      doi={}
}
```




## References
- Nathan Michael, Daniel Mellinger, Quentin Lindsey, Vijay Kumar (2010) [*The GRASP Multiple Micro UAV Testbed*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.169.1687&rep=rep1&type=pdf)
- Benoit Landry (2014) [*Planning and Control for Quadrotor Flight through Cluttered Environments*](http://groups.csail.mit.edu/robotics-center/public_papers/Landry15)
- Julian Forster (2015) [*System Identification of the Crazyflie 2.0 Nano Quadrocopter*](http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf)
- Carlos Luis and Jeroome Le Ny (2016) [*Design of a Trajectory Tracking Controller for a Nanoquadcopter*](https://arxiv.org/pdf/1608.05786.pdf)
- Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor (2017) [*AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles*](https://arxiv.org/pdf/1705.05065.pdf)
- Eric Liang, Richard Liaw, Philipp Moritz, Robert Nishihara, Roy Fox, Ken Goldberg, Joseph E. Gonzalez, Michael I. Jordan, and Ion Stoica (2018) [*RLlib: Abstractions for Distributed Reinforcement Learning*](https://arxiv.org/pdf/1712.09381.pdf)
- Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) [*Stable Baselines3*](https://github.com/DLR-RM/stable-baselines3)
- Guanya Shi, Xichen Shi, Michael O’Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung (2019)
[*Neural Lander: Stable Drone Landing Control Using Learned Dynamics*](https://arxiv.org/pdf/1811.08027.pdf)
- Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, and Shimon Whiteson (2019) [*The StarCraft Multi-Agent Challenge*](https://arxiv.org/pdf/1902.04043.pdf)
- C. Karen Liu and Dan Negrut (2020) [*The Role of Physics-Based Simulators in Robotics*](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-072220-093055)
- Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza (2020) [*Flightmare: A Flexible Quadrotor Simulator*](https://arxiv.org/pdf/2009.00563.pdf)






Bonus GIF for scrolling this far

<img src="files/readme_images/2020.gif" alt="formation flight" width="350"> <img src="files/readme_images/2020.png" alt="control info" width="450">





-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute) / University of Cambridge's [Prorok Lab](https://github.com/proroklab) / [Mitacs](https://www.mitacs.ca/en/projects/multi-agent-reinforcement-learning-decentralized-uavugv-cooperative-exploration)
