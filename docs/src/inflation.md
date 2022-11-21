# Inflation 
Inflation is an approach that slows down collapse in ensemble Kalman methods.
Two distinct forms of inflation are implemented in this package. Both involve perturbing the ensemble members following the standard update rule of the chosen Kalman process.
Multiplicative inflation expands ensemble members away from their mean in a
deterministic manner, whereas additive inflation hinges on the addition of stochastic noise to ensemble members.

For both implementations, a scaling factor ``s`` is included to extend functionality to cases with mini-batching. 
The scaling factor ``s`` multiplies the artificial time step ``\Delta t`` in the inflation equations to account for sampling error. For mini-batching, the scaling factor should be:
```math
    s = \frac{|B|}{|C|}
```
where `` |B| `` is the mini-batch size and `` |C| `` is the full dataset size.

## Multiplicative Inflation 
Multiplicative inflation effectively scales parameter vectors in parameter space, such that the perturbed
ensemble remains in the linear span of the original ensemble. The implemented update equation follows
[Huang et al, 2022](https://arxiv.org/abs/2204.04386) eqn. 41:

```math
\begin{aligned}
    m_{n+1} = m_{n} ; \qquad u^{j}_{n + 1} = m_{n+1} + \sqrt{\frac{1}{1 - s \Delta{t}}} \left(u^{j}_{n} - m_{n} \right) \qquad (1)
\end{aligned}
```
where ``m`` is the ensemble average. In this way,
the parameter covariance is inflated by a factor of ``\frac{1}{1 - s \Delta{t}}``, while the ensemble mean remains fixed.
```math
     C_{n + 1} = \frac{1}{1 - s \Delta{t}} C_{n} \qquad (2)
```

Multiplicative inflation can be used by flagging the `update_ensemble!` method as follows:
```julia
    EKP.update_ensemble!(ekiobj, g_ens; multiplicative_inflation = true, s = 1.0)
```

## Additive Inflation 
Additive inflation is implemented by systematically adding stochastic perturbations to the parameter ensemble in the form of Gaussian noise. Additive inflation breaks the linear subspace property, meaning the parameter ensemble can evolve outside of the span of the initial ensemble. In additive inflation, the ensemble is perturbed in the following manner after the standard Kalman update:

```math
     u_{n+1} = u_n + \zeta_{n} \qquad (3) \\
    \zeta_{n} \sim N(0, \frac{s \Delta{t} }{1 - s \Delta{t}} C_n) \qquad (4)
```
This inflates the parameter covariance by a factor of ``\frac{1}{1 - s \Delta{t}}`` as in eqn. 2 , while the ensemble mean remains fixed.

Additive inflation can be used by flagging the `update_ensemble!` method as follows:
```julia
    EKP.update_ensemble!(ekiobj, g_ens; additive_inflation = true, s = 1.0)
```
Alternatively, the prior covariance matrix may be used to generate additive noise, following:
```math
    \zeta_{n} \sim N(0, \frac{s \Delta{t} }{1 - s \Delta{t}} C_{0}) \qquad (5)
```
This results in an additive increase in the parameter covariance by `` \frac{s \Delta{t} }{1 - s \Delta{t}} * C_{0}`` , while the mean remains fixed.
```math
     C_{n + 1} = C_{n} + \frac{s \Delta{t} }{1 - s \Delta{t}} C_{0} \qquad (6)
```

Additive inflation using the scaled prior covariance (parameter covariance of initial ensemble) can be used by flagging the `update_ensemble!` method as follows:
```julia
    EKP.update_ensemble!(ekiobj, g_ens; additive_inflation = true, use_prior_cov = true, s = 1.0)
```