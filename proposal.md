# Cold-Start-Aware Extension of ComplexMatrixComb Using a Frozen Cell-Metadata Encoder and Two-Dimensional Coordinates

## 1. Introduction

Drug combination therapy is an important strategy in cancer treatment because two drugs can be administered together to obtain a desired inhibitory effect on cancer cells. A fundamental practical question is determining the concentrations of the two drugs required to achieve a particular level of inhibition.

Most computational drug-combination approaches formulate the problem in the forward direction:

$$
(\text{Cell Line},\text{Drug A},\text{Drug B},\text{Dose A},\text{Dose B})
\longrightarrow
\text{Inhibition}.
$$

Our previous work instead addressed the inverse problem:

$$
(\text{Cell Line},\text{Drug A},\text{Drug B},\text{Target Inhibition})
\longrightarrow
(\text{Dose A},\text{Dose B}).
$$

In other words, rather than asking what inhibition will result from a predefined pair of concentrations, the objective is to determine the concentrations of two drugs required to obtain a desired inhibitory effect.

Our previous work, **ComplexMatrixComb**, formulated this inverse drug-combination dose-prediction problem and focused particularly on identifying drug concentration pairs associated with approximately 50% inhibition.

The new project maintains this objective while addressing an important limitation of the previous approach: **the cell-line cold-start problem**.

The proposed architecture introduces exactly two conceptual modifications:

1. Cell metadata are first compressed using the encoder of a separately trained Variational Autoencoder. This pretrained encoder is frozen during training of the final model.

2. Complex numbers are replaced by an explicit two-dimensional coordinate representation. The two drug-associated quantities are treated as two coordinates rather than as the real and imaginary components of a complex number.

The remaining structure of the proposed architecture is preserved.

---

# 2. Previous Work: ComplexMatrixComb

## 2.1 Problem Addressed by the Previous Work

The objective of the previous work was to predict the concentrations of two drugs required to achieve a predefined inhibition level for a particular cancer cell line.

For a cell line \(c\) and drug pair

$$
(d_A,d_B),
$$

the objective was to estimate

$$
(\mu_A,\mu_B),
$$

where \(\mu_A\) and \(\mu_B\) are the concentrations of Drug A and Drug B associated with the target inhibitory effect.

The work particularly focused on approximately 50% inhibition.

Therefore, the problem was formulated as

$$
(c,d_A,d_B)
\longrightarrow
(\mu_A,\mu_B)
$$

for the target inhibition level.

This differs from conventional drug-response prediction, where concentrations are provided as inputs and inhibition is predicted as the output.

ComplexMatrixComb therefore directly addressed the inverse dose-prediction problem.

---

# 3. Method Used in the Previous Work

ComplexMatrixComb represented the concentrations of two drugs using a single complex-valued quantity.

For Drug A concentration \(\mu_A\) and Drug B concentration \(\mu_B\),

$$
y=\mu_A+i\mu_B.
$$

The concentration of one drug was therefore represented by the real component and the concentration of the other drug by the imaginary component.

The method consisted of the following principal steps:

1. scaling drug concentrations;
2. selecting observations corresponding to the target inhibition level;
3. constructing a cell-line/drug-pair matrix;
4. representing the two drug concentrations as the two components of a complex-valued matrix entry;
5. performing complex-valued matrix factorization;
6. predicting missing concentration pairs.

The previous method therefore constructed a matrix of the form

$$
CD\in\mathbb C^{m\times t},
$$

where

- \(m\) represents the number of cell lines;
- \(t\) represents the number of drug pairs.

Each observed matrix entry contained the two concentrations associated with the target inhibition level.

The matrix was approximated through a low-rank factorization involving latent representations for cell lines and drug pairs.

Conceptually,

$$
CD\approx CD_{\text{latent}}^T+B.
$$

After learning the latent factors, missing entries could be estimated, and their two components could be interpreted as the concentrations of the two drugs.

---

# 4. Limitation of the Previous Architecture: Cell-Line Cold Start

The matrix-factorization formulation has an important limitation.

The model learns a latent representation for every cell line appearing in the training matrix.

Suppose the training data contain the cell lines

$$
c_1,c_2,\ldots,c_m.
$$

The model learns corresponding latent vectors

$$
C_1,C_2,\ldots,C_m.
$$

Prediction for a cell line therefore depends on its learned latent representation.

For an entirely new cell line

$$
c_{\text{new}},
$$

which was not present during model training, there is no previously learned latent representation

$$
C_{\text{new}}.
$$

Consequently, the original matrix-factorization approach is naturally suited to completing missing observations involving entities represented during training, but it cannot directly construct a representation for a completely unseen cell line from biological information about that cell.

This is the **cell-line cold-start problem**.

The previous ComplexMatrixComb formulation is transductive and relies on learned cell-line and drug-pair latent representations, which creates this limitation for fully unseen entities.

The purpose of the new proposal is therefore to provide the architecture with a mechanism for constructing a representation from the **metadata of the cell line itself**.

---

# 5. Proposed Extension

The new architecture preserves the overall proposed model while making two modifications.

## Modification 1: Offline VAE Encoder for Cell Metadata

Raw cell metadata are no longer provided directly to the Cell Encoder.

Instead, a completely separate Variational Autoencoder is first trained using cell metadata.

The VAE consists of

$$
\text{VAE Encoder}
\rightarrow
\text{Latent Representation}
\rightarrow
\text{VAE Decoder}.
$$

After VAE training is completed:

1. the decoder is removed;
2. the encoder is retained;
3. the encoder is inserted before the Cell Encoder of the final architecture;
4. all parameters of the pretrained VAE encoder are frozen;
5. the VAE encoder is not updated during training of the final model.

The final cell-processing pipeline therefore becomes

$$
\text{Raw Cell Metadata}
\rightarrow
\boxed{\text{Frozen Pretrained VAE Encoder}}
\rightarrow
\text{Compressed Cell Metadata}
\rightarrow
\text{Cell Encoder}.
$$

The purpose of this component is to generate a compressed representation from the metadata of a cell line.

For an unseen cell line, its metadata can therefore be passed through the same frozen encoder:

$$
x_{\text{new}}
\rightarrow
f_{\text{VAE}}(x_{\text{new}})
\rightarrow
Z_{\text{cell,new}}.
$$

The final architecture consequently does not require the raw cell metadata to be provided directly to the Cell Encoder.

---

## Modification 2: Two-Dimensional Coordinates Instead of Complex Numbers

The second modification concerns the mathematical representation.

The original approach used complex-number terminology:

$$
a+ib,
$$

with one drug represented through the real component and the other through the imaginary component.

However, the proposed architecture does not require the defining complex-number property

$$
i^2=-1.
$$

The architecture only requires two values that remain distinguishable and exchange their positions when the two drugs are exchanged.

We therefore replace the complex representation with an explicit two-dimensional coordinate representation:

$$
\mathbf z=
\begin{bmatrix}
z_1\\
z_2
\end{bmatrix}.
$$

Instead of referring to

$$
Z_R,\quad Z_I,
$$

as real and imaginary components, we use

$$
Z_1,\quad Z_2,
$$

as the two coordinates.

Similarly, other quantities previously represented using real and imaginary labels are represented using coordinate labels \(1\) and \(2\).

Under exchange of Drug A and Drug B,

$$
(Z_1,Z_2)
\longrightarrow
(Z_2,Z_1).
$$

The architecture therefore maintains two separate drug-associated coordinates without treating them as a complex number.

---

# 6. Overall Architecture

```text
====================================================================================================

                 OFFLINE STAGE: CELL-METADATA VARIATIONAL AUTOENCODER

====================================================================================================


                              [Cell Metadata]
                                    |
                                    v
                         +----------------------+
                         |     VAE ENCODER      |
                         +----------------------+
                                    |
                                    v
                         [Compressed Cell Latent]
                                    |
                                    v
                         +----------------------+
                         |     VAE DECODER      |
                         +----------------------+
                                    |
                                    v
                        [Reconstructed Metadata]


                              VAE IS TRAINED


                                    |
                                    v


                         Remove VAE Decoder
                                +
                         Freeze VAE Encoder


====================================================================================================

                     FINAL MODEL USES ONLY THE FROZEN VAE ENCODER

====================================================================================================
```

The offline VAE is not part of the optimization procedure of the final architecture.

---

# 7. Phase 1: Input Stage

```text
====================================================================================================

                     PHASE 1: THE INPUT STAGE

====================================================================================================


 [Cell Metadata]          [Target Inhibition]      [Drug A Metadata]        [Drug B Metadata]
       |                         |                        |                        |
       v                         |                        |                        |
+------------------+             |                        |                        |
| FROZEN PRETRAINED|             |                        |                        |
|   VAE ENCODER    |             |                        |                        |
+------------------+             |                        |                        |
       |                         |                        |                        |
       v                         |                        |                        |
 [Z_cell]                       |                        |                        |
       |                         |                        |                        |
       v                         v                        v                        v
+--------------+         +--------------+      +------------------------------------------+
| Cell Encoder |         | Inhib Encoder|      |       SHARED DRUG ENCODER                |
|  Trainable   |         |  Real Val.   |      |              SIAMESE                      |
+--------------+         +--------------+      +------------------------------------------+
       |                         |                        |                        |
       v                         v                        v                        v
   [E_cell]                    [E_x]                   [E_drugA]                [E_drugB]
       |                         |                        |                        |
       +-------------------------+-----------+------------+------------------------+
                                             |

====================================================================================================
```

The only difference in cell processing relative to the original proposed architecture is the insertion of the pretrained frozen VAE encoder before the Cell Encoder.

The Cell Encoder itself remains part of the final trainable architecture.

---

# 8. Phase 2: The Two Parallel Paths

```text
====================================================================================================

                          PHASE 2: THE TWO "BRAINS" — PARALLEL PATHS

====================================================================================================

                                             |
              +------------------------------+-----------------------------+
              |                                                            |
              v                                                            v

+------------------------------------------+             +--------------------------------------+
| PATH A: TWO-COORDINATE TENSOR MF         |             | PATH B: DEEP NEURAL NETWORK          |
| Structured Factorization                 |             | Non-linear biological interactions   |
+------------------------------------------+             +--------------------------------------+
|                                          |             |                                      |
| 1. CELL & INHIBITION FACTORS             |             | 1. CONCATENATE FEATURES              |
|                                          |             |    SYMMETRICALLY                     |
| C_1 = W_C1(E_cell)                       |             |                                      |
| C_2 = W_C2(E_cell)                       |             | Cat_A = [E_cell,E_x,E_A,E_B]          |
|                                          |             |                                      |
| X_1 = W_X1(E_x)                          |             | Cat_B = [E_cell,E_x,E_B,E_A]          |
| X_2 = W_X2(E_x)                          |             |                                      |
|                                          |             |                                      |
| 2. TWO-COORDINATE DRUG-PAIR FACTOR       |             | 2. SHARED DEEP MLP                   |
|                                          |             |                                      |
| D_1 = W_D([E_A (+) E_B])                 |             | H_A = Deep_MLP(Cat_A)                 |
|                                          |             |                                      |
| D_2 = W_D([E_B (+) E_A])                 |             | H_B = Deep_MLP(Cat_B)                 |
|                                          |             |                                      |
| Same W_D is used for both.               |             | Same MLP is used for both.            |
|                                          |             |                                      |
| 3. CONTEXT MERGE                         |             | 3. COMBINE H_A AND H_B                |
|                                          |             |                                      |
| U_1 = C_1 (*) X_1                        |             | H_deep = [H_A (+) H_B]                |
|                                          |             |                                      |
| U_2 = C_2 (*) X_2                        |             |                                      |
|                                          |             |                                      |
| 4. TWO-COORDINATE TENSOR MULTIPLICATION  |             |                                      |
|                                          |             |                                      |
| Z_1 = (U_1 (*) D_1) + (U_2 (*) D_2)      |             |                                      |
|                                          |             |                                      |
| Z_2 = (U_1 (*) D_2) + (U_2 (*) D_1)      |             |                                      |
|                                          |             |                                      |
| Z_MF = [Z_1 (+) Z_2]                     |             |                                      |
+------------------------------------------+             +--------------------------------------+
              |                                                            |
              +------------------------------+-----------------------------+
                                             |

====================================================================================================
```

Legend:

- `(*)` = element-wise multiplication
- `(+)` = concatenation

---

# 9. Two-Dimensional Coordinate Interpretation

The tensor-factorization branch now produces

$$
Z_1,Z_2
$$

rather than real and imaginary components.

Define

$$
\mathbf Z=
\begin{bmatrix}
Z_1\\
Z_2
\end{bmatrix}.
$$

The ordered drug-pair factors are

$$
D_1=W_D(E_A\oplus E_B),
$$

$$
D_2=W_D(E_B\oplus E_A).
$$

When Drug A and Drug B are exchanged,

$$
E_A\leftrightarrow E_B,
$$

and therefore

$$
D_1\leftrightarrow D_2.
$$

The two output coordinates consequently exchange their positions:

$$
Z_1\leftrightarrow Z_2.
$$

The paired representation can therefore be viewed directly as a two-dimensional coordinate system:

$$
(Z_1,Z_2).
$$

No imaginary unit is required.

---

# 10. Phase 3: Output Stage

The output stage remains unchanged from the original proposed architecture.

```text
====================================================================================================

                            PHASE 3: THE OUTPUT STAGE

====================================================================================================

                                             |
                                             v

                              +-----------------------+
                              | FUSION CONCATENATION  |
                              | [Z_MF (+) H_deep]     |
                              +-----------------------+
                                             |
                                             v

                         +---------------------------------+
                         | MIXTURE DENSITY NETWORK (MDN)   |
                         | Outputs M Gaussian Peaks        |
                         +---------------------------------+
                           /              |              \
                          /               |               \
                         v                v                v

                 +------------+     +------------+    +-------------+
                 | MIX WEIGHTS|     |   MEANS    |    | VARIANCES   |
                 |    (pi)    |     |   (mu)     |    | (sigma,rho) |
                 |   [B,M]    |     | [B,M,2]    |    |  [B,M,3]    |
                 +------------+     +------------+    +-------------+
                                             |
                                             v

                          Generates 2D Isobologram Probability Plane

====================================================================================================
```

The two output dimensions correspond to the concentrations of Drug A and Drug B.

---

# 11. Exact Inputs, Outputs, and Dimensions

## Hyperparameter Definitions

- \(B\): batch size
- \(N_c\): number of raw cell-metadata features
- \(N_d\): number of raw drug-metadata features
- \(K\_{VAE}\): dimension of the pretrained VAE latent representation
- \(K\_{enc}\): dimension of the architecture's latent embeddings
- \(K\_{MF}\): factorization rank
- \(K\_{MLP}\): dimension of the deep-branch output
- \(M\): number of Gaussian mixture components

---

# 12. Offline VAE

## Input

$$
Raw\_Cell
\in
\mathbb R^{B\times N_c}.
$$

## Encoder

The VAE encoder compresses the cell metadata into its latent representation.

$$
Z_{\text{cell}}
\in
\mathbb R^{B\times K_{VAE}}.
$$

## Decoder

During standalone VAE training, the decoder reconstructs the original cell metadata from the latent representation.

After VAE training is completed, the decoder is removed.

Only the encoder is retained.

## Final Status

During training of the final architecture,

$$
\boxed{\text{VAE Encoder Parameters Are Frozen}}.
$$

The gradients produced by the final model do not update the VAE encoder.

---

# 13. Phase 1: Encoders

## 13.1 Frozen VAE Encoder

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

## 13.2 Cell Encoder

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

## 13.3 Inhibition Encoder

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

## 13.4 Siamese Drug Encoder

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

# 14. Phase 2, Path A: Two-Coordinate Tensor Matrix Factorization

## 14.1 Cell and Inhibition Factors

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

## 14.2 Two-Coordinate Drug-Pair Factor

Inputs:

$$
E_A,E_B.
$$

The two ordered concatenations are passed through the same linear layer \(W_D\):

$$
D_1
=
W_D(E_A\oplus E_B),
$$

$$
D_2
=
W_D(E_B\oplus E_A).
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

## 14.3 Context Merge

$$
U_1=C_1\odot X_1,
$$

$$
U_2=C_2\odot X_2.
$$

---

## 14.4 Two-Coordinate Tensor Interaction

$$
Z_1
=
(U_1\odot D_1)
+
(U_2\odot D_2),
$$

$$
Z_2
=
(U_1\odot D_2)
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
Z_{MF}
=
[Z_1\oplus Z_2]
\in
\mathbb R^{B\times2K_{MF}}.
$$

---

# 15. Phase 2, Path B: Deep Neural Network

## 15.1 Symmetric Concatenation

Inputs:

$$
E_{\text{cell}},
E_x,
E_A,
E_B.
$$

Construct

$$
Cat_A
=
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
Cat_B
=
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

## 15.2 Shared Deep MLP

Both representations are passed through exactly the same deep neural network:

$$
H_A
=
DeepMLP(Cat_A),
$$

$$
H_B
=
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
H_{\text{deep}}
=
[H_A\oplus H_B]
\in
\mathbb R^{B\times K_{MLP}}.
$$

---

# 16. Phase 3: Mixture Density Network

## 16.1 Fusion

Inputs:

$$
Z_{MF},H_{\text{deep}}.
$$

Concatenate:

$$
Fusion\_Vec
=
[Z_{MF}\oplus H_{\text{deep}}].
$$

Therefore,

$$
Fusion\_Vec
\in
\mathbb R^{B\times(2K_{MF}+K_{MLP})}.
$$

---

## 16.2 MDN

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

# 17. Drug-Order Behavior

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

The difference from the original complex-number interpretation is only that these two channels are now explicitly interpreted as coordinates:

$$
(Z_1,Z_2)
$$

rather than as

$$
Z_R+iZ_I.
$$

---

# 18. Complete Proposed Architecture

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

                              Coordinate 1 = Dose A
                              Coordinate 2 = Dose B

====================================================================================================
```

---

# 19. Final Output: Two-Dimensional Dose Probability Map

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

# 20. Summary of the Proposed Changes

The proposed model retains the architecture originally designed for the project.

Only two changes are introduced.

## Change 1 — Frozen Offline Cell-Metadata Encoder

Original:

$$
\text{Cell Metadata}
\rightarrow
\text{Cell Encoder}.
$$

Proposed:

$$
\boxed{
\text{Cell Metadata}
\rightarrow
\text{Frozen Pretrained VAE Encoder}
\rightarrow
\text{Cell Encoder}
}.
$$

The VAE is trained independently.

Its decoder is removed after training.

Its encoder is frozen during training of the final architecture.

This modification provides a feature-based compressed representation of the cell metadata before it enters the trainable Cell Encoder.

---

## Change 2 — Two Coordinates Instead of Complex Numbers

Original terminology:

$$
Z_R+iZ_I.
$$

Proposed terminology:

$$
\boxed{(Z_1,Z_2)}.
$$

The model therefore treats the two quantities as coordinates in a two-dimensional representation rather than as the real and imaginary components of a complex number.

There is no requirement for

$$
i^2=-1.
$$

The important structural property is that exchanging the two drugs exchanges the two associated coordinates.

---

# 21. Proposed Model in One Expression

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

The objective remains the same:

$$
\boxed{
(\text{Cell},\text{Drug A},\text{Drug B},\text{Target Inhibition})
\rightarrow
\text{Probability Distribution over }(\text{Dose A},\text{Dose B})
}.
$$

The proposal therefore extends the previous work toward cell-line cold-start prediction while preserving the originally proposed model structure and replacing complex-number notation with an explicit two-dimensional coordinate representation.
