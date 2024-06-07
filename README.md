# Make value functions more robust by using mixture of gaussians and reducing uncertainty

## Two approaches within this repository with regards to loss functions:
- Energy distance between target and predicted distributions
    - Source: https://arxiv.org/pdf/2105.11366.pdf
- Negative log likelihood  as the loss between target and predicted distributions
    1. Not verbatim, but Source: https://arxiv.org/pdf/2204.10256.pdf
    2. Source of loss function: rice-field
 ## Reducing uncertainty using Epistemic Neural Networks
 - Use of base networks, prior networks, and epistemic networks to capture epistemic uncertainty (lack of knowledge / exploration)
    1. ENN foundation source: https://arxiv.org/pdf/2107.08924.pdf
    2. ENN implementation into RL source: https://arxiv.org/pdf/2302.09205.pdf
    3. Prior network research source: https://arxiv.org/pdf/1806.03335
 - This implementation has been tried in a rigorous environment and performed better compared to a transformer with stacked frames
 - The ENNWrapper is an attempet at abstracting the ENN away from the model to allow for integration of any neural network
    - This requires a network to be passed to the wrapper
    - Specified z dimensions (think of this as the number of models in an ensemble -- this is a hyperparameter and between 5-15 has been effective)
    - Activation function to use (current supported are relu, LeakyReLU, and elu -- will be adding more)
    - enn layers for both the learnable network and the prior networks
    - hidden layer of the base network that is being passed to the wrapper
