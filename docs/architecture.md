# Model architecture

The model predicts a correlated distribution over two drug doses from cell metadata, shared drug features and target inhibition. A separately trained VAE supplies a frozen cell encoder; paired factorization and deep branches feed a bivariate mixture density head.

## 1. Exact Inputs, Outputs, and Dimensions

### Hyperparameter Definitions

- \(B\): batch size
- \(N_c\): number of raw cell-metadata features
- \(N_d\): number of raw drug-metadata features
- \(K\_{VAE}\): dimension of the pretrained VAE latent representation
- \(K\_{enc}\): dimension of the architecture's latent embeddings
- \(K\_{MF}\): factorization rank
- \(K\_{MLP}\): dimension of the deep-branch output
- \(M\): number of Gaussian mixture components

---

## 2. Offline VAE

### Input

$$
Raw\_Cell
\in
\mathbb R^{B\times N_c}.
$$

### Encoder

The VAE encoder compresses the cell metadata into its latent representation.

$$
Z_{\text{cell}}
\in
\mathbb R^{B\times K_{VAE}}.
$$

### Decoder

During standalone VAE training, the decoder reconstructs the original cell metadata from the latent representation.

After VAE training is completed, the decoder is removed.

Only the encoder is retained.

### Final Status

During training of the final architecture,

$$
\boxed{\text{VAE Encoder Parameters Are Frozen}}.
$$

The gradients produced by the final model do not update the VAE encoder.

---

## 3. Encoders

### 3.1 Frozen VAE Encoder

Input:

$$
Raw\_Cell
\in
\mathbb R^{B\times N_c}.
$$

Output:

$$
Z_{\text{cell}}
\in
\mathbb R^{B\times K_{VAE}}.
$$

---

### 3.2 Cell Encoder

Input:

$$
Z_{\text{cell}}
\in
\mathbb R^{B\times K_{VAE}}.
$$

Operation:

```text
Dense -> ReLU -> Dense
```

Output:

$$
E_{\text{cell}}
\in
\mathbb R^{B\times K_{enc}}.
$$

The Cell Encoder remains trainable.

---

### 3.3 Inhibition Encoder

Input:

$$
Raw_x
\in
\mathbb R^{B\times1}.
$$

Operation:

```text
Dense -> ReLU
```

Output:

$$
E_x
\in
\mathbb R^{B\times K_{enc}}.
$$

---

### 3.4 Siamese Drug Encoder

Inputs:

$$
Raw\_drugA
\in
\mathbb R^{B\times N_d},
$$

$$
Raw\_drugB
\in
\mathbb R^{B\times N_d}.
$$

Operation:

```text
Dense -> ReLU -> Dense
```

with exactly the same weights for the two drugs.

Outputs:

$$
E_A
\in
\mathbb R^{B\times K_{enc}},
$$

$$
E_B
\in
\mathbb R^{B\times K_{enc}}.
$$

---

## 4. Path A: Two-Coordinate Tensor Matrix Factorization

### 4.1 Cell and Inhibition Factors

Inputs:

$$
E_{\text{cell}},E_x.
$$

Four distinct linear layers produce

$$
C_1,C_2
\in
\mathbb R^{B\times K_{MF}},
$$

and

$$
X_1,X_2
\in
\mathbb R^{B\times K_{MF}}.
$$

Specifically,

$$
C_1=W_{C1}(E_{\text{cell}}),
$$

$$
C_2=W_{C2}(E_{\text{cell}}),
$$

$$
X_1=W_{X1}(E_x),
$$

$$
X_2=W_{X2}(E_x).
$$

---

### 4.2 Two-Coordinate Drug-Pair Factor

Inputs:

$$
E_A,E_B.
$$

The two ordered concatenations are passed through the same linear layer \(W_D\):

$$
D_1=W_D(E_A\oplus E_B),
$$

$$
D_2=W_D(E_B\oplus E_A).
$$

Therefore,

$$
D_1,D_2
\in
\mathbb R^{B\times K_{MF}}.
$$

When the two drugs exchange positions,

$$
D_1\leftrightarrow D_2.
$$

---

### 4.3 Context Merge

$$
U_1=C_1\odot X_1,
$$

$$
U_2=C_2\odot X_2.
$$

---

### 4.4 Two-Coordinate Tensor Interaction

$$
Z_1=(U_1\odot D_1)
+
(U_2\odot D_2),
$$

$$
Z_2=(U_1\odot D_2)
+
(U_2\odot D_1).
$$

Outputs:

$$
Z_1,Z_2
\in
\mathbb R^{B\times K_{MF}}.
$$

Flatten for fusion:

$$
Z_{MF}=
[Z_1\oplus Z_2]
\in
\mathbb R^{B\times2K_{MF}}.
$$

---

## 5. Path B: Deep Neural Network

### 5.1 Symmetric Concatenation

Inputs:

$$
E_{\text{cell}},
E_x,
E_A,
E_B.
$$

Construct

$$
Cat_A=
[E_{\text{cell}}
\oplus
E_x
\oplus
E_A
\oplus
E_B],
$$

and

$$
Cat_B=
[E_{\text{cell}}
\oplus
E_x
\oplus
E_B
\oplus
E_A].
$$

Therefore,

$$
Cat_A,Cat_B
\in
\mathbb R^{B\times4K_{enc}}.
$$

---

### 5.2 Shared Deep MLP

Both representations are passed through exactly the same deep neural network:

$$
H_A=
DeepMLP(Cat_A),
$$

$$
H_B=
DeepMLP(Cat_B).
$$

Outputs:

$$
H_A
\in
\mathbb R^{B\times(K_{MLP}/2)},
$$

$$
H_B
\in
\mathbb R^{B\times(K_{MLP}/2)}.
$$

Flatten for fusion:

$$
H_{\text{deep}}=
[H_A\oplus H_B]
\in
\mathbb R^{B\times K_{MLP}}.
$$

---

## 6. Mixture Density Network

### 6.1 Fusion

Inputs:

$$
Z_{MF},H_{\text{deep}}.
$$

Concatenate:

$$
Fusion\_Vec=
[Z_{MF}\oplus H_{\text{deep}}].
$$

Therefore,

$$
Fusion\_Vec
\in
\mathbb R^{B\times(2K_{MF}+K_{MLP})}.
$$

---

### 6.2 MDN

The Mixture Density Network maps the fused representation to parameters of a two-dimensional Gaussian mixture.

### Mixture Weights

$$
\pi
\in
\mathbb R^{B\times M}.
$$

The mixture weights are produced using a linear layer followed by Softmax.

---

### Means

$$
\mu
\in
\mathbb R^{B\times M\times2}.
$$

The two dimensions represent:

$$
(\text{Dose A},\text{Dose B}).
$$

---

### Variances and Correlation

$$
(\sigma,\rho)
\in
\mathbb R^{B\times M\times3}.
$$

The three parameters correspond to

$$
\sigma_A,
\quad
\sigma_B,
\quad
\rho.
$$

Softplus is applied to the standard-deviation parameters and tanh is applied to the correlation parameter.

---

## 7. Drug-Order Behavior

The architecture processes the two drugs through the same Siamese Drug Encoder.

Therefore, when

$$
Drug_A\leftrightarrow Drug_B,
$$

the embeddings satisfy

$$
E_A\leftrightarrow E_B.
$$

In Path A,

$$
D_1\leftrightarrow D_2,
$$

which leads to

$$
Z_1\leftrightarrow Z_2.
$$

In Path B,

$$
Cat_A\leftrightarrow Cat_B,
$$

and because both are processed by the same MLP,

$$
H_A\leftrightarrow H_B.
$$

The paired structure of the representation is therefore preserved when the drug ordering is exchanged.

The paired channels are the coordinates $(Z_1,Z_2)$.

---

## 8. Complete Architecture

```text
====================================================================================================

                            OFFLINE TRAINING

====================================================================================================

                         [Cell Metadata Dataset]
                                  |
                                  v
                        +-------------------+
                        | VARIATIONAL       |
                        | AUTOENCODER       |
                        +-------------------+
                         /                 \
                        v                   v
                 VAE Encoder            VAE Decoder
                        |                   |
                        +-------------------+
                                  |
                              Train VAE


                         AFTER VAE TRAINING:

                         Keep Encoder
                         Remove Decoder
                         Freeze Encoder


====================================================================================================

                             FINAL ARCHITECTURE

====================================================================================================


 [Cell Metadata]       [Target Inhibition]       [Drug A Metadata]       [Drug B Metadata]
       |                       |                        |                       |
       v                       |                        |                       |
+------------------+           |                        |                       |
| FROZEN VAE       |           |                        |                       |
| ENCODER          |           |                        |                       |
+------------------+           |                        |                       |
       |                       |                        |                       |
       v                       |                        |                       |
   [Z_cell]                    |                        |                       |
       |                       |                        |                       |
       v                       v                        v                       v
+--------------+       +--------------+       +-----------------------------------------+
| Cell Encoder |       |Inhib Encoder |       |        SHARED DRUG ENCODER              |
+--------------+       +--------------+       |             SIAMESE                      |
       |                       |              +-----------------------------------------+
       v                       v                        |                       |
   [E_cell]                  [E_x]                    [E_A]                   [E_B]
       |                       |                        |                       |
       +-----------------------+-------------+----------+-----------------------+
                                             |
                    +------------------------+------------------------+
                    |                                                 |
                    v                                                 v

       +--------------------------------+              +--------------------------------+
       | PATH A: TWO-COORDINATE        |              | PATH B: DEEP NEURAL NETWORK   |
       | TENSOR MATRIX FACTORIZATION    |              |                                |
       +--------------------------------+              +--------------------------------+
                    |                                                 |
                 Z_1,Z_2                                           H_A,H_B
                    |                                                 |
                    +------------------------+------------------------+
                                             |
                                             v
                              +-----------------------------+
                              |           FUSION            |
                              | [Z_1,Z_2,H_A,H_B]           |
                              +-----------------------------+
                                             |
                                             v
                              +-----------------------------+
                              | MIXTURE DENSITY NETWORK     |
                              +-----------------------------+
                                  /          |          \
                                 v           v           v
                               pi           mu      sigma_A,
                                                   sigma_B,rho
                                             |
                                             v

                            2D DRUG-DOSE PROBABILITY MAP

                              Coordinate 1 Dose A
                              Coordinate 2 = Dose B

====================================================================================================
```

---

## 9. Final Output: Two-Dimensional Dose Probability Map

The final model represents possible concentration pairs in an explicit two-dimensional coordinate system.

```text
==========================================================================================

                        FINAL OUTPUT: THE 2D PROBABILITY MAP

==========================================================================================

   Dose B
     ^

     |                       ***
High |                     **   **
     |                     *  x  *
     |                     **   **
     |                       ***

     |

     |                                    ***
Low  |                                  **   **
     |                                  *  x  *
     |                                  **   **
     |                                    ***

     +----------------------------------------------------> Dose A

                            Low                   High

==========================================================================================
```

Each point in the plane is represented as

$$
(d_A,d_B).
$$

The MDN models a probability distribution over this two-dimensional dose space.

---

## 10. Model in One Expression

The resulting model can be summarized as

$$
\boxed{
\begin{aligned}
\text{Cell Metadata}
&\xrightarrow{\text{Frozen Offline VAE Encoder}}
Z_{\text{cell}}
\xrightarrow{\text{Cell Encoder}}
E_{\text{cell}},\\[4pt]
\text{Target Inhibition}
&\xrightarrow{\text{Inhibition Encoder}}
E_x,\\[4pt]
Drug_A
&\xrightarrow{\text{Shared Drug Encoder}}
E_A,\\[4pt]
Drug_B
&\xrightarrow{\text{Shared Drug Encoder}}
E_B,\\[4pt]
(E_{\text{cell}},E_x,E_A,E_B)
&\xrightarrow{\text{Two-Coordinate MF + Deep Path}}
(Z_1,Z_2,H_A,H_B),\\[4pt]
(Z_1,Z_2,H_A,H_B)
&\xrightarrow{\text{Fusion + MDN}}
p(d_A,d_B).
\end{aligned}
}
$$

The prediction objective is:

$$
\boxed{
(\text{Cell},\text{Drug A},\text{Drug B},\text{Target Inhibition})
\rightarrow
\text{Probability Distribution over }(\text{Dose A},\text{Dose B})
}.
$$
