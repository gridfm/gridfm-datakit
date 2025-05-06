# Branch Current Computation in Power Grids

We consider a power network with:

- $n$ buses,  
- $m$ branches (lines or transformers),  
- complex bus voltages $V \in \mathbb{C}^n$ in per-unit,  
- admittance values $y \in \mathbb{C}^m$ associated with each branch.

Each branch $b \in \{1, \dots, m\}$ connects a *from-bus* $\sigma_f(b)$ to a *to-bus* $\sigma_t(b)$. The series admittance matrix for branch $b$ is defined as:

$$
\begin{bmatrix}
I_{f_b} \\
I_{t_b}
\end{bmatrix}
=
\begin{bmatrix}
y_{ff_b} & y_{ft_b} \\
y_{tf_b} & y_{tt_b}
\end{bmatrix}
\begin{bmatrix}
V_{\sigma_f(b)} \\
V_{\sigma_t(b)}
\end{bmatrix}
$$

where:

- $I_{f_b}$ and $I_{t_b}$ are the complex currents injected into the branch from the from-bus and to-bus, respectively,  
- $y_{ff_b}, y_{ft_b}, y_{tf_b}, y_{tt_b}$ are complex coefficients representing the branch admittance model.

## Matrix Formulation

We define the branch current injection matrix $Y_f \in \mathbb{C}^{m \times n}$ such that:

$$
I_f = Y_f V
$$

where the $b$-th row of $Y_f$ is:

$$
(Y_f)_{b,:} = e_{\sigma_f(b)}^\top y_{ff_b} + e_{\sigma_t(b)}^\top y_{ft_b}
$$

with $e_i \in \mathbb{R}^n$ the $i$-th unit basis vector. Similarly, we define $Y_t$ for the to-end currents:

$$
I_t = Y_t V
$$

## Conversion to Physical Units (kA)

To convert the per-unit current $I_f$ to physical units (kA), we use the base apparent power $S_{\text{base}}$ (in MVA) and base voltage $V_{\text{base}}$ (in kV):

$$
(I_{f,\text{kA}})_b = \frac{I_{f_b} \cdot S_{\text{base}}}{\sqrt{3} \cdot V_{\text{base}_{\sigma_f(b)}}}
$$

This gives the phase current magnitude in kiloamperes under balanced conditions. 

Similarly, we have:
$$
(I_{t,\text{kA}})_b = \frac{I_{t_b} \cdot S_{\text{base}}}{\sqrt{3} \cdot V_{\text{base}_{\sigma_t(b)}}}
$$

## Thermal Limits and Loading

Let $\text{limit}_f \in \mathbb{R}^m$ be the thermal current limit (in kA) on each branch from-end. We define the branch loading as:

$$
\text{loading}_b = \frac{\max(|(I_{f,\text{kA}})_b|, |(I_{t,\text{kA}})_b|)}{\text{limit}_f(b)}
$$

Lines usually connect buses with the same voltage level, so that $(I_{f,\text{kA}})_b = (I_{t,\text{kA}})_b$, which is not the case for transformers.
