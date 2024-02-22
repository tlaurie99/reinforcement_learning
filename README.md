# Make value functions more robust by using mixture of gaussians

## Two approaches within this repository:
- Energy distance between target and predicted distributions
            - Source: https://arxiv.org/pdf/2105.11366.pdf
- Negative log likelihood  as the loss between target and predicted distributions
            1. Not verbatim, but Source: https://arxiv.org/pdf/2204.10256.pdf
            2. Source of loss function: rice-field
