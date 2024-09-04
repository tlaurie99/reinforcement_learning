# Improving the current reinforcement learning paradigm one neuron at a time

## Testing and improving loss functions:
- Energy distance between target and predicted distributions
    - Source: https://arxiv.org/pdf/2105.11366.pdf
- Negative log likelihood  as the loss between target and predicted distributions
    1. Not verbatim, but Source: https://arxiv.org/pdf/2204.10256.pdf
    2. Source of loss function: rice-field
 ## Reducing epistemic uncertainty using Epistemic Neural Networks
 - Base knowledge: epistemic uncertainty is lack of knowledge while aleatoric uncertainty is due to ambiguity
 - Use of base networks, prior networks, and epistemic networks to capture epistemic uncertainty (lack of knowledge / exploration)
    1. ENN foundation source: https://arxiv.org/pdf/2107.08924.pdf
    2. ENN implementation into RL source: https://arxiv.org/pdf/2302.09205.pdf
    3. Prior network research source: https://arxiv.org/pdf/1806.03335
 - This implementation has been tried in a rigorous environment and performed better compared to a transformer with stacked frames
 - The ENNWrapper is an attempt at abstracting the ENN away from the model to allow for integration of any neural network
    - This requires a network to be passed to the wrapper
    - Specified z dimensions (think of this as the number of models in an ensemble -- this is a hyperparameter and between 5-15 has been effective)
    - Activation function to use
    - enn layers for both the learnable network and the prior networks
    - hidden layer of the base network that is being passed to the wrapper
    - Additional ability to calculate loss for the base network and the ENN network (MoG networks as well -- though this has had not good performance in testing)
- ENNWrapper has increased the learning ability significantly of agents in dog fighting scenarious (using PyFlyt). Within 1M timesteps the agent is completing dominating (~20,000 mean reward vs ~-1,000 mean reward)
      by the end of the simulation (30M timesteps) the agent reaches 70,000+ mean reward and the other agent falls to -2,000+ mean reward.
    - This simulation was tested 30+ individual times with the same hyperparameters, reversing agent order, instantiating differently within the env, etc.
    - Currently tesing the effect on increasing the Z-dim (brief results have shown that 10-15 is the optimal vs 1-10)
  
