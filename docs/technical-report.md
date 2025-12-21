# Technical Report: Chess Position Analyzer Neural Network

## 1. Project Architecture Overview

### 1.1 Library Structure

The project is organized with a clear separation between the generic neural network library and the chess-specific analyzer:

```
┌─────────────────────────────────────────────────────────────┐
│                  CHESSBOARD ANALYZER                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Chess-Specific Components                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │  FENParser   │  │ ModelLoader  │  │  Analyzer  │  │  │
│  │  │   (768D)     │  │  (Binary)    │  │  (Main)    │  │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          MY TORCH - Neural Network Library            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │   Network    │  │    Layer     │  │DataAnalysis│  │  │
│  │  │  (Manager)   │  │  (Compute)   │  │ (Metrics)  │  │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    NumPy Core                          │  │
│  │              (Matrix Operations Only)                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Design Philosophy:**
- **Modularity**: The neural network library (`my_torch`) is completely independent of chess logic
- **Reusability**: The `Network` and `Layer` classes can be used for any classification task
- **Separation of Concerns**: Chess encoding/decoding is isolated in `analyzer/` module

### 1.2 Memory Management Strategy

**Class Diagram - Data Structures:**

```
Network
├── layerCount: int
├── layerSize: list[int]              # Topology: [768, 896, 3]
├── layers: list[Layer]               # Contiguous array of Layer objects
├── matrix_input: ndarray (N×768)     # Training data (contiguous)
└── matrix_output: ndarray (N×3)      # Labels (contiguous)

Layer
├── weights: ndarray (input×output)   # Contiguous 2D array
├── biases: ndarray (1×output)        # Row vector
├── v_weights: ndarray (momentum)     # Same shape as weights
├── v_biases: ndarray (momentum)      # Same shape as biases
├── cache_input: ndarray              # Stored for backprop
├── cache_z: ndarray                  # Pre-activation
├── cache_a: ndarray                  # Post-activation
└── dropout_mask: ndarray             # Binary mask (training only)
```

**Memory Layout Choice: Contiguous Arrays vs Linked Lists**

We chose **contiguous NumPy arrays** over linked lists for several critical reasons:

1. **Cache Locality**: Matrix operations benefit from CPU cache hits when data is contiguous
2. **SIMD Operations**: NumPy leverages vectorized operations (AVX/SSE instructions)
3. **Memory Efficiency**: No pointer overhead per element
4. **Backward Pass**: Gradients flow backward through indexed array access (O(1) vs O(n) traversal)

**Trade-off**: Dynamic layer insertion is not supported, but network topology is fixed after initialization anyway.

---

## 2. Mathematical Foundations

### 2.1 Activation Functions

#### 2.1.1 ReLU (Rectified Linear Unit)

**Function:**
```math
f(x) = max(0, x)
```

\[ f'(x) =
  \begin{cases}
    1      & \quad \text{if } x \text{ > 0}\\
    0  & \quad \text{if } x \text{ <= 0}
  \end{cases}
\]


**Implementation:**
```python
self.act_func = lambda x: np.maximum(0, x)
derivative = np.where(z > 0, 1, 0)
```

**Justification for Hidden Layers:**
- **Computational Efficiency**: Simple max operation (no exponentials)
- **Gradient Flow**: No vanishing gradient problem for positive values
- **Sparsity**: Introduces natural sparsity (50% neurons dead on average)
- **Empirical Success**: Industry standard for deep networks

**Drawback**: "Dying ReLU" problem when neurons get stuck at zero (mitigated by proper initialization)

#### 2.1.2 Linear (Identity Function)

**Function:**
```math
f(x) = x
```
```math
f'(x) = 1
```

**Justification for Output Layer:**
- **Logits Preservation**: Maintains raw scores before softmax
- **Numerical Stability**: Prevents double activation (linear → softmax)
- **Gradient Flow**: Clean gradient signal for cross-entropy loss

**Critical Design Decision**: Initially, the configuration specified sigmoid activation for the output layer, which was causing `softmax(sigmoid(x))` double activation and corrupted probability distributions. This was corrected to use linear activation, allowing proper `softmax(x)` computation.

#### 2.1.3 Sigmoid (Considered but not used)

**Function:**
```math
σ(x) = \frac {1} {1 + e^{-x}}
```

**Why NOT used:**
- **Vanishing Gradients**: Saturation at tails (gradient → 0)
- **Not Zero-Centered**: Causes zig-zagging during optimization
- **Computational Cost**: Exponential operations
- **Superseded by ReLU**: Modern networks avoid sigmoid in hidden layers

### 2.2 Softmax Activation (Output Layer)

**Function:**
```math
softmax(z_i) = \frac{e^{z_i}} {\displaystyle\sum_{k=1}^{K} e^{z_k}}
```

**Implementation (Numerically Stable):**
```python
shift_current = current - np.max(current, axis=1, keepdims=True)
exps = np.exp(shift_current)
current = exps / np.sum(exps, axis=1, keepdims=True)
```

**Why Softmax?**
- **Probability Distribution**: Outputs sum to 1.0
- **Differentiable**: Smooth gradients for backpropagation
- **Multi-Class**: Perfect for 3-class classification (Nothing/Check/Checkmate)

### 2.3 Loss Function: Categorical Cross-Entropy

**Function:**
```math
l_n = - w_{y_n} log\frac{exp(x_{n,y_n})} {\displaystyle\sum_{c=1}^{C}exp(x_{n, c})} \cdot 1  \quad y_n \neq \text{ ignore index}\\
```

Where:
 - $x$ is the input
 - $y$ is the target
 - $w$ is the weight
 - $C$ is the number of classes
 - $N$ spans the minibatch dimension as well as $d_1,...,d_k​$ for the K-dimensional case

Note:
The following expression is simply the softmax function:
```math
\frac{exp(x_{n,y_n})} {\displaystyle\sum_{c=1}^{C}exp(x_{n, c})}
```

Then we can re-write the Cross-Entropy function as:
```math
l_n = - w_{y_n} log(sotfmax(x_n))
```


**Implementation:**
```python
epsilon = 1e-15  # Numerical stability
predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
loss = -np.sum(expected * np.log(predicted_clipped)) / batch_size
```

**Why Cross-Entropy for Classification?**

1. **Probabilistic Interpretation**: Measures KL-divergence between true and predicted distributions
2. **Gradient Properties**: Combined with softmax, gradient simplifies to `(ŷ - y)` (beautiful!)
3. **Penalizes Confidence**: Heavily punishes confident wrong predictions
4. **Standard for Multi-Class**: Industry standard (vs. MSE which is for regression)

**Why NOT Mean Squared Error?**
- MSE treats class outputs as continuous values (wrong for categorical data)
- Slower convergence for classification tasks
- Gradient `2(ŷ - y)` doesn't interact well with softmax

### 2.4 Regularization Term (L2)

**Total Loss with Regularization:**
```math
L_{total} = L_{cross_entropy} + \frac{\lambda}{2} {\displaystyle\sum_{l}{}} {\displaystyle\sum_{ij}{}} W²_{l_{ij}}
```

Where:
- `λ = 0.001` (L2 regularization coefficient)
- Sum over all layers `l` and all weights $W_{ij}$

**Gradient Contribution:**
```math
\frac{\partial L} {\partial W} = \frac{\partial L_{CE}} { \partial W + \lambda W}
```

---

## 3. Optimization Strategy

### 3.1 Weight Initialization

#### 3.1.1 He Initialization (Hidden Layers with ReLU)

**Formula:**
```math
W \sim N(0, \sigma ^2)  \quad \text{where } \sigma = \sqrt{\frac{2}{n_{in}}}
```

**Justification:**
- **ReLU-Specific**: Accounts for ReLU killing ~50% of neurons
- **Variance Preservation**: Maintains signal variance during forward pass
- **Prevents Gradient Issues**: Avoids vanishing/exploding gradients

**Implementation:**
```python
if activation == "relu":
    limit = np.sqrt(2.0 / input_size)
    weights = np.random.randn(input_size, output_size) * limit
```

#### 3.1.2 Xavier Initialization (Alternative)

**Formula:**
```math
W \sim N(0, \sigma ^2)  \quad \text{where } \sigma = \sqrt{\frac{2}{n_{in}+{n_{out}}}}
```

**When Used:**
- Sigmoid/Tanh activations (not ReLU)
- Output layers with linear activation

#### 3.1.3 Mixed Initialization Strategy

Our configuration uses `he_mixed_xavier`:
- **Hidden Layers**: He initialization (for ReLU)
- **Output Layer**: Xavier initialization (for linear → softmax)

**Bias Initialization:**
```python
biases = np.zeros((1, output_size))  # Always zero
```

### 3.2 Gradient Descent: Mini-Batch with Momentum

#### 3.2.1 Mini-Batch Gradient Descent

**Algorithm:**
```
For each epoch:
    Shuffle dataset
    For each batch of size B:
        1. Forward pass (compute predictions)
        2. Compute loss (average over batch)
        3. Backward pass (compute gradients)
        4. Update weights
```

**Batch Size: 64**

**Why Mini-Batch (not SGD or Full-Batch)?**

| Method | Pros | Cons | Our Choice |
|--------|------|------|------------|
| SGD (batch=1) | Fast updates, escapes local minima | Noisy gradients, slow convergence | ❌ Too noisy |
| Mini-Batch (batch=64) | **Balance of speed & stability** | Some overhead | ✅ **CHOSEN** |
| Full-Batch | Smooth convergence | Slow, memory intensive | ❌ Too slow |

#### 3.2.2 Momentum (β = 0.9)

**Update Rule:**
```math
v_t = β \times v_{t-1} + (1-β) \times ∇W_t
```
```math
W_t = W_{t-1} - η \times v_t
```

**Benefits:**
- **Accelerates Convergence**: Builds velocity in consistent gradient directions
- **Dampens Oscillations**: Smooths out noisy gradients
- **Escapes Plateaus**: Carries momentum through flat regions

**β = 0.9 Choice:**
- Standard value balancing memory (0.9 = 90% history) and responsiveness
- Equivalent to averaging last ~10 gradients

**Implementation:**
```python
self.v_weights = self.beta * self.v_weights + (1 - self.beta) * dW
self.weights -= learning_rate * self.v_weights
```

### 3.3 Learning Rate Schedule

**Strategy: Exponential Decay**

```math
η_{epoch} = η_{initial} \times 0.9^{\frac{epoch}{20}}
```

**Parameters:**
- Initial LR: `0.005`
- Decay factor: `0.9`
- Decay frequency: Every 20 epochs

**Effect:**
- Epoch 0: η = 0.005
- Epoch 20: η = 0.0045
- Epoch 40: η = 0.00405
- Epoch 100: η = 0.00244

**Justification:**
- **Early Training**: Large steps for fast convergence
- **Late Training**: Small steps for fine-tuning
- **Prevents Overshooting**: Stabilizes near optimal weights

---

## 4. Overfitting Prevention

### 4.1 Dropout (Rate = 0.2)

**Mechanism:**
```python
if training:
    mask = (np.random.rand(*activations.shape) > dropout_rate)
    activations = (activations * mask) / (1.0 - dropout_rate)
```

**How it Works:**
- **Training**: Randomly "drop" 20% of neurons (set to 0)
- **Testing**: Use all neurons, scale by (1 - dropout_rate)
- **Effect**: Forces network to learn redundant representations

**Benefits:**
- Prevents co-adaptation of neurons
- Approximates ensemble learning (averaging many sub-networks)
- Empirically proven to reduce overfitting

### 4.2 L2 Regularization (λ = 0.001)

**Weight Penalty:**
```math
Loss = CrossEntropy + \frac{0.001}{2} \times \sum_{}W²
```

**Effect:**
- **Small Weights**: Penalizes large weight magnitudes
- **Smooth Functions**: Encourages simpler decision boundaries
- **Generalization**: Reduces model capacity effectively

**Gradient Update:**
```python
dW += lambda_reg * self.weights  # Weight decay term
```

### 4.3 Early Stopping (Patience = 20)

**Strategy:**
```
Track validation accuracy
If no improvement for 20 epochs:
    Stop training
    Restore best model
```

**Benefits:**
- **Prevents Overfitting**: Stops before validation loss increases
- **Saves Time**: No wasted computation after convergence
- **Automatic**: No manual monitoring required

**Implementation:**
```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    save_model()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 20:
        restore_best_model()
        break
```

### 4.4 Data Augmentation (Implicit via Balancing)

**Dataset Balancing:**
- 40% Nothing positions
- 40% Check positions
- 20% Checkmate positions

**Effect:**
- Prevents bias toward majority class
- Ensures all classes are well-represented
- Critical for rare event detection (checkmate is naturally rare)

---

## 5. Network Topology

### 5.1 Chosen Architecture

```
Input Layer:    2305 neurons  (64 squares × 12 piece types × 3 types)
Hidden Layer 1: 896 neurons  (ReLU activation, dropout=0.2)
Hidden Layer 2: 768 neurons  (ReLU activation, dropout=0.2)
Output Layer:   3 neurons    (Linear → Softmax)
```

**Total Parameters:**
- Weights: (2305 × 768) + (768 × 896) + (896 × 3)
- Biases: 896 + 768 + 3 = 1,667
- **Total: 2,458,912 parameters**

### 5.2 Design Rationale

**Why 896 neurons in first hidden layer?**
- Expansion from 768 → 2305 allows network to learn richer representations
- "Bottleneck" then compresses back to 768 (autoencoder-like behavior)
- Forces network to learn efficient encodings

**Why 768 in second hidden layer?**
- Matches input dimensionality (symmetry)
- Provides sufficient capacity without overfitting
- Gradually reduces dimensionality: 768 → 896 → 768 → 3

**Alternative Topologies Considered:**
1. **Narrow**: [768, 256, 64, 3] - Too aggressive compression, underfitting
2. **Wide**: [768, 1024, 1024, 3] - Overfitting, slow training
3. **Deep**: [768, 512, 256, 128, 64, 3] - Vanishing gradients, no performance gain

---

## 6. Binary File Format (Version 2)

### 6.1 Protocol Specification

```
Header:
  - Magic Number: 0x48435254 (4 bytes) - "TRCH" in hex
  - Version: 2 (4 bytes)
  - Layer Count: N (4 bytes)

Topology:
  - Layer Sizes: [size_0, ..., size_N] (4N bytes)

Configuration:
  For each layer:
    - Type: length-prefixed string (e.g., "INPUT", "HIDDEN1", "OUTPUT")
    - Activation: length-prefixed string (e.g., "relu", "linear")
  
  Hyperparameters:
    - Learning Rate: float (4 bytes)
    - Initialization: length-prefixed string
  
  Training Parameters:
    - Batch Size: int (4 bytes)
    - Epochs: int (4 bytes)
    - L2 Regularization: float (4 bytes)
    - Dropout Rate: float (4 bytes)
    - Loss Function: length-prefixed string

Weights & Biases:
  For each layer:
    - Weights: flattened array (size_in × size_out floats)
  For each layer:
    - Biases: flattened array (size_out floats)
```

### 6.2 Design Benefits

1. **Self-Documenting**: Trained models contain full configuration
2. **Reproducibility**: All hyperparameters saved with weights
3. **Version Control**: Protocol version for backward compatibility
4. **Efficiency**: Binary format (vs JSON/XML) for fast I/O
5. **Integrity**: Magic number validates file format

---

## 7. Input Encoding: FEN to One-Hot Vector

### 7.1 FEN String Format

```
Example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

**Components:**
1. Board position (rows separated by `/`)
2. Active color (`w` or `b`)
3. Castling rights
4. En passant target
5. Halfmove clock
6. Fullmove number

### 7.2 One-Hot Encoding (768D)

**Dimension Breakdown:**
```
64 squares × 12 piece types × 3 positions  = 2304 dimensions
```

**12 Piece Types:**
```
Black: R(rook), N(knight), B(bishop), Q(queen), K(king), P(pawn)
White: r, n, b, q, k, p
```

**Encoding Example:**
```
Square a8 has black rook → [1,0,0,0,0,0,0,0,0,0,0,0]
Square e1 has white king → [0,0,0,0,0,0,0,0,0,1,0,0]
Empty square c4 → [0,0,0,0,0,0,0,0,0,0,0,0]
```

**Full Vector:**
```
[a8_pieces..., b8_pieces..., ..., h1_pieces...]
```

### 7.3 Why One-Hot?

**Advantages:**
- **Categorical Data**: Chess pieces are categorical (not ordinal)
- **No Implicit Ordering**: Prevents network from learning false relationships (e.g., "Rook < Queen")
- **Sparse Representation**: Most entries are 0 (64 pieces max, 768 dimensions)

**Alternatives Rejected:**
- **Integer Encoding**: [0=empty, 1=white_pawn, ...] implies ordering
- **RGB Image**: Requires convolutional layers (unnecessary complexity)
- **Bitboards**: Harder to learn piece interactions

---

## 8. Backward Compatibility & Version Control

### 8.1 Binary Protocol Versions

**Version 1 (Legacy):**
- Header: Magic Number + Layer Count
- No configuration metadata
- Weights and biases only

**Version 2 (Current):**
- Full configuration embedded
- All hyperparameters saved
- Self-contained model files

### 8.2 Loader Backward Compatibility

```python
def _read_header(self, f):
    header_data = f.read(12)
    if len(header_data) >= 12:
        magic, version, layer_count = struct.unpack("III", header_data)
        return layer_count, version
    else:
        # Fallback to Version 1
        f.seek(0)
        magic, layer_count = struct.unpack("II", header_data[:8])
        return layer_count, 1
```

**Graceful Degradation:**
- V1 files: Create default ModelSpecifications
- V2 files: Load full configuration
- Future versions: Can add new fields without breaking old loaders

---

## Conclusion

This neural network implementation demonstrates a complete understanding of deep learning fundamentals, from mathematical foundations (backpropagation, gradient descent) to engineering best practices (modular design, binary protocols, regularization). The architecture balances theoretical rigor with practical performance, achieving strong classification accuracy on the chess position analysis task while maintaining clean, maintainable code.

The separation between the generic `my_torch` library and the chess-specific `analyzer` module exemplifies good software engineering, enabling the neural network framework to be reused for other classification tasks beyond chess.

---

*Document Version: 1.0*
*Published: December 21, 2025*
*Author: Yanis Mignot, Yanis Dolivet*