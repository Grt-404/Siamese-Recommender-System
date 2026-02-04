# Mentor Compatibility Recommendation System 

## Overview

The goal of this challenge is to predict pairwise compatibility between participants of networking events using profile attributes.

Each participant is described by:

• Numerical features (Age, Company Size)  
• Categorical features (Role, Seniority, Industry, Location)  
• Textual features (Business Interests, Objectives, Constraints)

Given historical compatibility scores for some user pairs, we must predict compatibility for all unseen pairs.

This is fundamentally a **pairwise matching problem**, not classical regression.

---

## Initial Intuition

My first instinct was to model this problem as **content-based filtering**, similar to collaborative filtering used in recommender systems.

The original idea:

1. Encode each user into a vector V
2. Predict compatibility using a dot product:

score = VA · VB


This approach assumes that compatibility is purely similarity-based.

However, during experimentation and analysis, several limitations became apparent:

• Dot product alone is too rigid  
• Cannot capture complementary relationships  
• Forces symmetric similarity only  
• Lacks expressive power for complex interactions  

This dataset is synthetic and does not follow real-world collaborative filtering assumptions. Therefore, a more flexible model was required.

---

## Key Realization

This is not traditional recommendation.

This is a **metric learning / pairwise matching problem**.

We are learning a function:

f(UserA, UserB) → Compatibility


This naturally leads to a **Siamese Neural Network** architecture.

---

## Final Architecture: Siamese Recommender Network

Each user profile is passed through a shared encoder network:

Profile → Encoder → Embedding (64D)


This produces two embeddings:

VA and VB

Instead of using only dot product, richer interaction features are constructed:

• VA  
• VB  
• |VA − VB| (absolute difference)  
• VA · VB (dot similarity)

These are concatenated and fed into a second neural network to predict compatibility.

This allows the model to learn:

• similarity  
• complementarity  
• asymmetric relationships  

This architecture is inspired by metric learning systems used in:

• recommender systems  
• matchmaking platforms  
• face recognition  
• semantic similarity models  

---

## Feature Engineering

Each user profile is converted into a single vector by concatenating:

1. Standardized numeric features
2. One-hot encoded categorical features
3. TF-IDF encoded text features

This avoids manual feature crafting and allows the neural network to learn representations directly.

---

## Training Strategy

### Symmetry Augmentation

Compatibility is symmetric.

So each pair (A,B) is mirrored as (B,A).

This doubles training data and enforces symmetry.

---

### Negative Sampling

The dataset provides only positive examples.

To prevent the model from predicting everything as compatible, synthetic negative pairs are created by randomly pairing users.

These negatives are assigned label 0.

This converts the problem into supervised binary classification.

---

## Loss Function Evolution

Initially Mean Squared Error was used.

However this caused:

• prediction collapse  
• very narrow output range  
• poor ranking behavior  

The problem is fundamentally classification/ranking, not regression.

Therefore Binary Cross Entropy was adopted.

This significantly improved:

• gradient signal  
• output spread  
• ranking capability  

---

## Model Stabilization Techniques

Several engineering improvements were required:

### Layer Normalization

Applied to embeddings to stabilize scale and prevent collapse.

### Absolute Difference Feature

Helps the model understand distance between embeddings.

### Dot Product Feature

Provides similarity signal directly.

### Dropout

Reduces overfitting.

---

## Engineering Challenges Faced

### 1. Dataset Only Provided Positive Examples

Solved using negative sampling.

---

### 2. Output Collapse (All Predictions Same)

Caused by MSE loss and lack of interaction features.

Solved using:

• Binary cross entropy  
• Dot product  
• Difference features  

---

### 3. Extremely Slow Pair Generation

Originally nested loops and numpy.vstack caused O(N²) memory copying.

Solved using vectorized numpy operations:

np.repeat
np.tile


This reduced generation time from minutes to seconds.

---

### 4. Model Serialization Issues

Lambda layers prevented safe deserialization.

Resolved by replacing TensorFlow ops with Keras layers.

---

## Final System

The final system consists of:

1. Profile encoder (shared Siamese network)
2. Pairwise interaction layer
3. Scoring network
4. Full cartesian prediction over test profiles

The output is a CSV containing:

ID = src_user_id + "_" + dst_user_id
compatibility_score


---

## Why This Approach

Tree-based models were avoided because:

• They cannot learn embeddings  
• They do not generalize to unseen pairs  
• They do not scale to cartesian prediction  

Siamese networks allow:

• representation learning  
• scalable inference  
• meaningful similarity modeling  

This mirrors production recommender architectures.

---

## Conclusion

The final model is a metric-learning based Siamese recommender system that learns latent representations of participants and predicts compatibility using learned interaction features.

The approach prioritizes architectural reasoning and learning dynamics over brute-force regression.

This design generalizes beyond this dataset and can directly support real-world matchmaking systems.

---
