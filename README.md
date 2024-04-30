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
