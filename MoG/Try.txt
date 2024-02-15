Try:
--clamping the exp term like so
first_term = sqrt_2_over_pi * torch.sqrt(sum_vars) * torch.exp((-diff_mu**2) / (2*sum_vars))
where I could do exp_term = torch.clamp(((-diff_mu**2) / (2*sum_vars)), -1e-5, 1e4)
--since this term was shown to be -1e-9
--->epiphany: this is **more than likely** from the internal dispersion terms since they are becoming very small overtime and causing **underflow**!
--->as the distributions become tighter and less variance & bias is involved their internal dispersion will be very small causing issues with the backward pass
Try:
--applying torch.abs(diff_mu) when doing torch.log(first_term)
Try:
--the log trick where you can apply an offset after doing torch.log(torch.abs(diff_mu)) + torch.log(added_value)
Try:
--work the math to try some sort of log_sumexp trick
