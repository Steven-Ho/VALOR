# VALOR - Variational Option Discovery Algorithms
This is a pytorch implementation of VALOR.

## Motivation of this project
Variational methods are recently introduced into reinforcement learning research. It allows RL algorithms learn various modes of policies besides maximize accumulated return. Mutual information measures the degree of relation between pre-sampled policy label and the following states or trajectories. By maximizing MI, we can assign $\pi$ different task labels, leading to different behavior.
VALOR is one of these variational methods that concentrates on MI between task labels and the whole trajectory - not states alone. The paper was [published](https://arxiv.org/abs/1807.10299). Unlike traditional RL procedure, a decoder is trained to recognize what the underlying task label is in collected trajectories. Furthermore, the performance of the decoder is added to reward function. Thus, the policy and the decoder should collaborate to learn a good way to transmit the information from pre-sampled label to the decoder.
Unfortunately, so far was no code released. This project partially implements this algorithm, including VPG learning procedure and MI maximization through a decoder (discriminator in the codes). The curriculum learning mechanism is omitted. hyperparameters are not carefully finetuned so you can reach a higher score on mujoco tasks.

## Prerequisites
OpenAI Gym
MUJOCO License
Pytorch

## Let it run!
Just run valor.py
