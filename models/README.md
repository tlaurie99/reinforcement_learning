# H1 MOG model
* MOG model uses the [MOG module / wrapper](../blob/main/MoG/MoG_module.py)


# H1 CBP model
* uses the idea from [Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7 "LOP/CBP")

# H1 clamped model
* has gained the attention of the RLLIB folks and is no longer needed -- instead log_std_clip_param can now be used as discussed [here](https://discuss.ray.io/t/ppo-nan-in-actor-logits/15140/7)
