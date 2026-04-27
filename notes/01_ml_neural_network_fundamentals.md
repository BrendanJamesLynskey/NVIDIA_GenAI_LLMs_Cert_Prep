# ML and Neural Network Fundamentals

The NCA-GENL exam weights "Core Machine Learning and AI Knowledge" at 30% — the single largest domain. It does not test introductory theory for its own sake; it assumes you can reason clearly about why training behaves as it does and what goes wrong when it does not. This note covers the building blocks the exam takes for granted: learning paradigms, the gradient-descent family, regularisation, data splits, activation functions, and initialisation. The LLM-specific lens — how these choices manifest inside a transformer — is in [notes/02\_transformer\_architecture.md](02_transformer_architecture.md) and in depth in [LLM\_Hub\_Transformer\_Architecture](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture).

---

## Learning paradigms and loss functions

The exam distinguishes three learning settings by what supervision signal is available.

**Supervised learning** pairs each input with a target label. The model minimises a loss that measures the distance between its prediction and the target. Common choices:

- *Cross-entropy* — the standard for classification. For a softmax output vector **p** and one-hot target **y**, the loss is −∑ yᵢ log pᵢ. Language model pre-training uses this with the next token as the target (teacher forcing).
- *Mean squared error (MSE)* — the natural choice for regression. Sensitive to outliers because squared errors penalise large deviations disproportionately.
- *KL divergence* — used when the target is a probability distribution rather than a hard label; appears in knowledge distillation and in the RLHF KL penalty term.

**Unsupervised learning** has no explicit labels. Objectives are constructed from the data itself — reconstruction error in autoencoders, contrastive objectives in representation learning (InfoNCE), or next-token prediction in self-supervised language modelling. The boundary with supervised learning is blurred in practice: GPT-style pre-training is technically self-supervised but optimises a cross-entropy loss against the next token.

**Reinforcement learning** replaces a fixed dataset with an agent that takes actions in an environment and receives scalar rewards. The policy gradient theorem (REINFORCE, PPO) optimises expected cumulative reward. In the LLM context this surfaces as RLHF: the environment is the conversation, the action is the next token, and the reward comes from a trained reward model. Training instability and reward hacking are the characteristic failure modes.

Exam distractor: "unsupervised and self-supervised are the same thing" — they are not. Self-supervised constructs a pseudo-label from the data; strictly unsupervised objectives (clustering, density estimation) do not.

---

## Forward pass, backprop, and optimisers

A neural network is a composition of differentiable operations. Given input **x**, the forward pass computes **ŷ = f(x; θ)**, then the loss L(ŷ, y). Backpropagation applies the chain rule in reverse through the computational graph to produce ∂L/∂θ for every parameter. This is automatic differentiation — PyTorch and JAX do it via reverse-mode AD.

**SGD** updates θ ← θ − η·g, where g is the minibatch gradient. It converges but is sensitive to learning rate and has poor behaviour in ravines (oscillates across the narrow dimension, advances slowly along the long one).

**Adam** maintains a per-parameter exponential moving average of both the gradient (first moment, m) and the squared gradient (second moment, v), and uses these to form an adaptive per-parameter step size. The update is approximately θ ← θ − η·m̂/(√v̂ + ε), where m̂ and v̂ are bias-corrected. The β₁ and β₂ hyperparameters (defaults 0.9 and 0.999) control the decay of each moment. Adam reaches good solutions faster than SGD on most deep-learning tasks.

**AdamW** decouples weight decay from the gradient update. Vanilla Adam applies L₂ regularisation by adding λθ to the gradient before the moment update — this means the effective decay rate is modulated by the adaptive step size, which reduces its regularising effect. AdamW instead applies the decay directly: θ ← (1 − ηλ)θ − η·m̂/(√v̂ + ε). This restores weight decay to its intended role and is the default optimiser for transformer training. The distinction matters on the exam.

**Learning rate scheduling** is standard practice. Warmup — linearly increasing η for the first few thousand steps — stabilises early training when the moment estimates are unreliable. Cosine annealing then decays η smoothly. A flat then cosine-decayed schedule (sometimes called "cosine with warmup") is near-universal in LLM training.

---

## Regularisation

Over-reliance on individual weights, or on the training set specifically, leads to poor generalisation. Regularisation techniques reduce this risk.

**L₂ weight decay** adds λ‖θ‖² to the loss, penalising large weights. It is equivalent to a Gaussian prior on weights. In practice, AdamW's decoupled form is preferred.

**L₁ regularisation** adds λ‖θ‖₁, which induces sparsity — many weights are pushed exactly to zero. Rarely used in transformers; more common in linear models and feature selection.

**Dropout** randomly zeroes activations during training with probability p (typically 0.1–0.5). At inference the activations are scaled by (1 − p) to compensate (or the weights are pre-scaled — "inverted dropout"). Dropout acts as an ensemble of exponentially many sub-networks. In modern transformer training it is applied sparingly — often only in the FFN layer — because it can interact badly with very large batch sizes and can slow convergence.

**Label smoothing** replaces hard one-hot targets with a softened distribution: (1 − ε) for the correct class and ε/(K − 1) for the remaining K − 1 classes. It prevents the model from becoming overconfident and improves calibration. It is equivalent to adding a uniform distribution term to the cross-entropy loss. The value ε = 0.1 is conventional.

**Early stopping** monitors validation loss and halts training when it begins to rise, regardless of training loss trajectory. It is computationally cheap and highly effective. The risk is stopping in a local optimum; in practice a patience window (e.g. "stop if no improvement for 5 evaluation steps") is used.

**Gradient clipping** is technically a training stability measure rather than regularisation, but it appears in the same toolbox. It caps the global gradient norm to a threshold (e.g. 1.0), preventing parameter updates from becoming catastrophically large. It is standard in RNN and transformer training.

---

## Train, validation, and test splits; diagnosing fit

The split exists to get unbiased performance estimates at each stage of development:

- **Training set** — the data the optimiser actually sees. Minimising training loss on this set is the objective.
- **Validation set** — held out during training, used for hyperparameter selection and early stopping. Because you make decisions based on it, the validation loss is a slightly optimistic estimate of generalisation.
- **Test set** — used once, to report final performance. Any tuning after seeing test-set results invalidates the estimate.

**k-fold cross-validation** partitions the data into k subsets, trains k models each using one different fold as validation, and averages the results. It gives a lower-variance estimate of generalisation error at the cost of k training runs. Preferred when data is scarce; less common in LLM training where datasets are large.

Diagnosing fit from loss curves:

| Pattern | Training loss | Validation loss | Diagnosis |
|---|---|---|---|
| Underfitting | High | High | Model too simple, or too few epochs |
| Overfitting | Low | Rising or stagnant | Model too complex, or dataset too small |
| Good fit | Declining | Declining, close to training | Generalisation is working |
| Distribution shift | Low | Persistently worse | Validation and training distributions differ |

The gap between training and validation loss is more diagnostic than either in isolation.

---

## Activation functions

Activations introduce the non-linearity that allows networks to learn non-trivial functions. The choice of activation has consequences for training dynamics at depth.

**ReLU** (Rectified Linear Unit): f(x) = max(0, x). Computationally cheap; does not saturate for positive inputs, so gradients flow. The "dying ReLU" problem occurs when a neuron consistently receives negative pre-activations — its gradient is zero and it stops learning. Kaiming initialisation (see below) was specifically motivated by ReLU.

**GELU** (Gaussian Error Linear Unit): f(x) = x · Φ(x), where Φ is the standard normal CDF. Approximated in practice as x · σ(1.702x) or x · 0.5 · (1 + tanh(√(2/π)(x + 0.044715x³))). Unlike ReLU, GELU is smooth and differentiable everywhere, it does not hard-zero negative inputs but stochastically gates them in proportion to the input magnitude, and it has a slight non-monotonicity near x ≈ −0.17. Empirically GELU outperforms ReLU in transformers. Used in BERT, GPT-2, and most subsequent models.

**SiLU / Swish**: f(x) = x · σ(x), where σ is the sigmoid. Resembles GELU but is simpler to compute. Used in LLaMA and its derivatives as part of the SwiGLU formulation (a gated linear unit variant of SiLU).

**SwiGLU**: Not a single activation but a gated FFN structure: output = (W₁x · SiLU(W₂x)) · W₃. It splits the FFN into two parallel linear projections, gates one with SiLU, and multiplies. Used in LLaMA, Mistral, Gemma. Because the gating doubles the linear projections, the inner dimension is conventionally reduced (e.g. from 4d_model to ~(8/3)d_model) to keep parameter count comparable to a standard FFN.

Why GELU/SwiGLU dominate in transformers: smooth gradients improve optimisation; the stochastic gating behaviour provides implicit regularisation; empirically they converge to lower perplexity than ReLU on language tasks. The difference is not a matter of principle but of empirical evidence across many training runs.

---

## Weight initialisation

Initialisation matters because poorly scaled weights cause activations to either saturate or collapse to zero before training begins, making gradients uninformative. At depth this effect compounds across layers.

**Xavier / Glorot initialisation** (Glorot and Bengio, 2010): scales initial weights so that activation variance is approximately preserved across layers for linear activations and for tanh. For a layer with nᵢₙ inputs and nₒᵤₜ outputs, weights are drawn from Uniform(−√(6/(nᵢₙ + nₒᵤₜ)), +√(6/(nᵢₙ + nₒᵤₜ))) or equivalently from a Normal with variance 2/(nᵢₙ + nₒᵤₜ). Works well with tanh and sigmoid activations. Default in many frameworks for linear and embedding layers.

**Kaiming / He initialisation** (He et al., 2015): accounts for ReLU's asymmetry — because ReLU zeros half the inputs, Xavier underestimates the required scale. Kaiming uses variance 2/nᵢₙ (for "fan-in mode"), correctly preserving variance through ReLU layers. The practical rule: use Kaiming for layers followed by ReLU or its variants; use Xavier for linear/tanh. PyTorch's `torch.nn.init.kaiming_normal_` defaults to "fan-in" mode with ReLU.

For transformers specifically, standard practice is to initialise attention projection weights and embedding matrices with a small normal distribution (σ ≈ 0.02 is common, derived from the GPT-2 paper) and to use a residual scaling factor of 1/√(2N) applied to the output projection of each sub-layer, where N is the number of layers. This prevents the residual stream variance from growing with depth.

---

## Likely exam angles

- **AdamW vs Adam**: The exam frequently tests whether you know what "decoupled weight decay" means. The correct answer is that Adam incorrectly applies weight decay through the adaptive learning rate, which reduces its effectiveness; AdamW applies it directly to the parameters before the gradient update.
- **Cross-entropy vs MSE for classification**: MSE is technically usable but cross-entropy is preferred because it is the proper scoring rule for probability outputs and its gradients are better-conditioned near saturation.
- **Dropout at inference**: Dropout is disabled at test time; activations are scaled by (1 − p) to keep expected values consistent. A common distractor claims the scaling happens at training time — in inverted dropout it does, but the result is the same.
- **Validation set contamination**: Any hyperparameter decision made using validation set performance means the validation set is no longer a clean generalisation estimate. The test set must remain unseen until final evaluation.
- **GELU vs ReLU in transformers**: GELU dominates not for theoretical reasons but empirical ones. The exam may test whether you know it is differentiable everywhere and has a smooth near-zero response, unlike ReLU's hard zero.
- **Kaiming vs Xavier**: Kaiming is preferred for ReLU/GELU activations because it accounts for the fact that roughly half of inputs are zeroed. Xavier was derived assuming linear activations and underestimates variance for asymmetric activations.

---

## Further reading

- Vaswani et al., "Attention Is All You Need" (2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Glorot and Bengio, "Understanding the Difficulty of Training Deep Feedforward Neural Networks" (2010): [http://proceedings.mlr.press/v9/glorot10a.html](http://proceedings.mlr.press/v9/glorot10a.html)
- He et al., "Delving Deep into Rectifiers" (2015): [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
- Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)" (2016): [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)
- Loshchilov and Hutter, "Decoupled Weight Decay Regularisation" (AdamW, 2017): [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- Noam et al., "The Transformer (nanoGPT walkthrough)" — [LLM\_Hub\_Transformer\_Architecture](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture) for the LLM-specific application of these fundamentals.
