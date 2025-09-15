---
title: part02_FM
toc_min_heading_level: 2
toc_max_heading_level: 3
---

# Flow Matching: A Deep Dive

---

## Normalizing Flows

### Overview
<details>
<summary>Overview</summary>

<div style={{textAlign: 'center'}}>

![Normalizing Flow Transformation](/img/normalizing_flow.png)

*Fig: A normalizing flow transforms a simple distribution into a complex one through a series of invertible mappings.*
</div>

- **Core Idea**: Transform a simple base distribution, $\pi(z_0)$ (e.g., standard Gaussian), into a complex target distribution, $\pi(z_K)$, by applying a sequence of invertible and differentiable mappings:

  $$
  z_0 \sim \pi(z_0) \quad \xrightarrow{f_1} \quad z_1 \xrightarrow{f_2} \cdots \xrightarrow{f_K} \quad z_K
  $$

  where each $f_i$ is an invertible transformation, and $z_K$ represents a sample from the learned, complex data distribution.

</details>

### Change of Variables Theorem
<details>
<summary>Change of Variables Theorem</summary>

If $z \sim \pi(z)$ and $x = f(z)$, where $f$ is invertible and differentiable, then the density of $x$ is:

$$
p(x) = \pi(z) \left| \det \frac{dz}{dx} \right| = \pi(f^{-1}(x)) \left| \det \frac{d f^{-1}(x)}{dx} \right|
$$

where $z = f^{-1}(x)$. This formula expresses how the probability density transforms from $z$ to $x$ under the mapping $x = f(z)$.

</details>

### Likelihood Calculation
<details>
<summary>Likelihood Calculation</summary>

To compute the likelihood of a data point $z_i$ after applying an invertible transformation $f_i$, we use the change of variable theorem:

$$
\mathbf{z}_{i-1} \sim p_{i-1}(\mathbf{z}_{i-1}) \\
\mathbf{z}_i = f_i(\mathbf{z}_{i-1}) \;\; \Longrightarrow \;\; \mathbf{z}_{i-1} = f_i^{-1}(\mathbf{z}_i) \\
p_i(\mathbf{z}_i) = p_{i-1}(f_i^{-1}(\mathbf{z}_i)) \left| \det \frac{d f_i^{-1}}{d \mathbf{z}_i} \right| = p_{i-1}(\mathbf{z}_{i-1}) \left| \det \frac{d f_i}{d \mathbf{z}_{i-1}} \right|^{-1} \\
\log p_i(\mathbf{z}_i) = \log p_{i-1}(\mathbf{z}_{i-1}) - \log \left| \det \frac{d f_i}{d \mathbf{z}_{i-1}} \right|
$$

**Stacked Transformations**:

$$
\mathbf{x} = \mathbf{z}_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0) \\
\log p(\mathbf{x}) = \log p_k(\mathbf{z_k}) = \log p_0(\mathbf{z}_0) - \sum_{i=1}^K \log \left| \det \frac{d f_i}{d \mathbf{z}_{i-1}} \right|
$$

**Negative log-likelihood** over the dataset $\mathcal{D}$:

$$
\mathcal{L}(\mathcal{D}) = -\frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \log p(\mathbf{x})
$$

</details>

### Practical Requirements and Implementations
<details>
<summary>Practical Requirements and Implementations</summary>

Each transformation $f_i$ should be:
- **Easily invertible**: Computing $f_i^{-1}$ should be efficient and tractable.
- **Efficient Jacobian determinant computation**: It should be easy to compute $\left| \det \frac{d f_i}{d \mathbf{z}_{i-1}} \right|$ for likelihood evaluation.

Common implementations that meet these criteria include: **NICE**, **RealNVP**, **Glow**, **MAF (Masked Autoregressive Flow)**.

</details>

### References
<details>
<summary>References</summary>

- [Lilian Weng's post on Flow Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
- [Deep Generative Models: Flow-based Models Notes](https://deepgenerativemodels.github.io/notes/flow/)

</details>


## Continuous Normalizing Flows (CNFs)

### CNF Solutions to NFs Limitations
<details>
<summary>CNF Solutions to Normalizing Flow Limitations</summary>

- **Continuous-time dynamics**: Model transformations via ODE-driven flows, eliminating the need for discrete, strictly invertible layers.

- **Efficient likelihood computation**: Replace log-determinant of Jacobian with time-integrated trace of the Jacobian, reducing computation cost.

- **Effective unlimited depth**: Use continuous-time evolution instead of stacking many discrete layers, avoiding depth-related overheads.

- **Flexible architectures**: Parameterize the vector field with arbitrary neural networks, removing strict invertibility constraints and specialized coupling designs.

</details>

### Mathematical Formulation of CNFs
<details>
<summary>Mathematical Formulation of CNFs</summary>

The key insight behind CNFs comes from the connection to **Residual Networks**. Consider the discrete update rule in ResNets:

$$
z_{t+1} = z_t + f(z_t)
$$

This is actually the **discrete form of the Euler solver** for an ordinary differential equation! By making the time step infinitesimally small, we can generalize this to the continuous case.

**From Discrete to Continuous**: The discrete ResNet update naturally leads to the continuous ODE formulation:

$$
\frac{dz_t}{dt} = f(z_t, t), \quad z_0 \sim p(z_0)
$$

This is the fundamental equation that defines how the latent variable $z(t)$ evolves continuously over time $t \in [0, T]$.

**Forward Transformation**: Given a latent variable $z_0$ with distribution $p(z_0)$, the transformation to the data space is defined as the solution to the ODE:

$$
z_T = z_0 + \int_0^T f(z_t, t) \, dt
$$

where $f(z_t, t)$ is a neural network that defines the vector field.

**Log-Likelihood Computation**: The log-likelihood of $x$ can be computed using the continuous change of variables formula. The key insight is that the change in log-probability is given by the divergence of the vector field (proof in appendix A of [the paper](https://arxiv.org/pdf/1806.07366)):

$$
\frac{d \log p(z_t)}{dt} = -\text{Tr}\left( \frac{\partial f}{\partial z_t} \right)
$$

Integrating this over time gives us the log-likelihood:

$$
\log p(z_T) = \log p(z_0) - \int_0^T \text{Tr}\left( \frac{\partial f}{\partial z_t} \right) dt
$$

This formulation allows for efficient computation of the log-determinant of the Jacobian through the trace, facilitating scalable and invertible density estimation.

</details>

### Key Advantages and Challenges
<details>
<summary>Key Advantages and Challenges</summary>

**Key advantages**
- **Conceptual Leap**: CNFs generalize discrete normalizing flows by defining the transformation as a **continuous-time process**.
- **Mechanism**: Instead of a sequence of layers, a CNF uses a **vector field** parameterized by a neural network to define the transformation. This vector field specifies the instantaneous rate of change of the data.
- **Mathematical Formulation**: The transformation is the solution to an **Ordinary Differential Equation (ODE)**. The path from a noise vector `z` to a data vector `x` is defined by integrating the vector field over time.
- **Advantage over Discrete Flows**: The change in log-probability is determined by the integral of the **trace of the Jacobian**, which is often much more computationally efficient than computing the determinant of the Jacobian for very deep, discrete models.

**Challenges**
- **Training Complexity**: Training CNFs often requires solving the ODE during the training process, which can be slow and computationally intensive.
- **ODE Solver Requirements**: The quality of the learned model depends on the accuracy of the ODE solver used during training and inference.

</details>

### References
<details>
<summary>References</summary>

- [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366)

</details>

## Flow Matching

### Overview
<details>
<summary>Flow Matching: A New Paradigm for Training CNFs</summary>

<div style={{textAlign: 'center'}}>

![Flow Matching](/img/flow_matching.png)

*Fig: Flow Matching Process. [Source](https://arxiv.org/pdf/2412.06264)*

</div>

- **Flow Matching (FM)** is a **simulation-free approach** for training Continuous Normalizing Flows (CNFs).
- **No ODE simulation or likelihood computation during training**: Instead of simulating ODE paths or computing likelihoods, FM directly regresses onto vector fields that generate the desired probability paths.
- **Key innovation**: FM frames training as a **direct regression problem**—learning the vector field $v_t$ that transforms noise into data, bypassing the need for expensive maximum likelihood estimation.
- **Main challenge**: The ideal ("marginal") vector field for the whole data distribution is **intractable** to compute, since it depends on the unknown, evolving probability distributions at every time step.
- **Solution**: FM introduces **Conditional Flow Matching (CFM)**, which makes training tractable by conditioning regression targets on individual data points.

</details>

### Mathematical Formulation

<details>
<summary>Flow Matching Objective and Flow ODE</summary>

<div style={{textAlign: 'center'}}>

![Vector Field](/img/vector_field.png)

*Visualization of vector fields and flow. [Source](https://arxiv.org/pdf/2412.06264)*

</div>

**Goal**: Learn a vector field $v_t(x; \theta)$ parameterized by a neural network that generates a probability path $p_t$ from a simple prior $p_0$ to the data distribution $p_1$.

**Flow ODE**: The probability path $p_t$ is generated by the flow of the vector field $v_t(x; \theta)$ through the ordinary differential equation:

$$
\frac{d\phi_t(x)}{dt} = v_t(\phi_t(x); \theta)
$$

where:
- $\phi_t(x)$ is the time-dependent map (called the flow), representing the state at time $t \in [0,1]$; in CNFs, this corresponds to $z_t$
- $v_t(\phi_t(x); \theta)$ is the learned vector field parameterized by $\theta$, corresponding to $f(z, t)$ in CNFs
- The flow transforms samples from $p_0$ to $p_1$ by integrating this ODE

**Flow Matching Loss**: The objective is to regress the neural network $v_t(x; \theta)$ onto a target vector field $u_t(x)$:

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, p_t(x)} \|v_t(x; \theta) - u_t(x)\|^2
$$

where:
- $t \sim \mathcal{U}(0,1)$ is sampled uniformly
- $u_t(x)$ is the target vector field that generates the desired probability path

**The Problem**: The target vector field $u_t(x)$ is intractable because it depends on the unknown marginal probability path $p_t(x)$.

</details>

### Conditional Flow Matching (CFM)
<details>
<summary>Conditional Flow Matching: Tractable Training with Optimal Transport Paths</summary>

**The Key Insight**: Instead of trying to learn the intractable marginal vector field $u_t(x)$, CFM defines **conditional probability paths** $p_t(x|x_1)$ and **conditional vector fields** $u_t(x|x_1)$ that are conditioned on individual data points $x_1$ from the training set.

**Conditional Flow Matching Loss**: The CFM objective is:

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)} \|v_t(x; \theta) - u_t(x|x_1)\|^2
$$

where:
- $t \sim \mathcal{U}(0,1)$ is sampled uniformly
- $x_1 \sim q(x_1)$ is sampled from the data distribution
- $u_t(x|x_1)$ is the conditional target vector field

**Optimal Transport Flow**: Instead of using complex diffusion paths, we can choose the **Optimal Transport (OT) displacement interpolation** as our conditional probability path:

$$
\phi_t(x|x_1) = (1-t)x_0 + tx_1
$$

This defines a **straight line path** from noise $x_0$ to data $x_1$.

**Conditional Vector Field**: The target vector field for OT paths is:

$$
\frac{d\phi_t(x|x_1)}{dt} = u_t(x|x_1) = x_1 - x_0
$$

**Simplified CFM Loss**: Substituting the OT vector field into the CFM objective:

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p_0(x_0)} \|v_t((1-t)x_0 + tx_1; \theta) - (x_1 - x_0)\|^2
$$

This is the **final, practical training objective** that can be computed efficiently without any ODE simulations!

**Advantages of OT Paths**:

**Straight Line Trajectories**: Unlike diffusion paths that follow curved trajectories, OT paths provide **straight line interpolation** between noise and data, leading to:
- **Faster training**: More direct optimization landscape
- **Faster sampling**: Fewer function evaluations needed
- **Better generalization**: Simpler paths are easier to learn

</details>

### Training Process
<details>
<summary>How to Train Flow Matching Models</summary>

**Training Algorithm**:
1. Sample a data point $x_1 \sim q(x_1)$ from the training set
2. Sample noise $x_0 \sim p_0(x_0)$ (e.g., standard Gaussian)
3. Sample time $t \sim \mathcal{U}(0,1)$ uniformly
4. Compute the interpolated point: $x_t = (1-t)x_0 + tx_1$
5. Compute the target vector: $u_t = x_1 - x_0$
6. Train the neural network to predict: $v_t(x_t; \theta) \approx u_t$
7. Update parameters using the CFM loss

**Key Advantages**:
- **No ODE simulation required** during training
- **Simple regression objective** - just predict the direction from noise to data
- **Stable training** - no likelihood computation or complex gradients

</details>

### Sampling Process
<details>
<summary>How to Generate Samples</summary>

**Sampling Algorithm**:
1. Start with noise: $x_0 \sim p_0(x_0)$
2. Solve the ODE from $t=0$ to $t=1$:

$$
\frac{dx_t}{dt} = v_t(x_t; \theta)
$$

3. Use any ODE solver (e.g., Euler, Runge-Kutta, adaptive solvers)
4. The final point $x_1$ is a sample from the learned distribution

**Sampling Efficiency**:
- **Fewer function evaluations** needed compared to diffusion models
- **Straight line paths** in OT case lead to faster convergence
- **Adaptive solvers** can automatically adjust step sizes for efficiency

**Example with Euler Solver**:
```python
# Pseudocode for sampling
x = sample_noise()  # x_0 ~ p_0
dt = 0.01  # time step
for t in range(0, 1, dt):
    v = neural_network(x, t)  # v_t(x; θ)
    x = x + dt * v  # Euler step
return x  # x_1 ~ p_1
```

</details>

### References

<details>
<summary>References</summary>

- [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)

</details>
