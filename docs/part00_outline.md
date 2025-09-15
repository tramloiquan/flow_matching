---
id: intro
title: part00_outline
sidebar_position: 1
slug: /
---

# Outline

This documentation provides a comprehensive overview of modern generative modeling, with a special focus on Flow Matching and its state-of-the-art applications.

## [Part 1: A Summary of Generative Modeling Techniques](./part01GM_overview)

This section provides a high-level summary of popular generative modeling techniques, setting the stage for understanding the advantages of Flow Matching.

- **Generative Adversarial Networks (GANs)**: Exploring the adversarial training process.
- **Variational Autoencoders (VAEs)**: Understanding probabilistic encoding and decoding.
- **Diffusion Models**: A deep dive into the denoising-based generation process.
- **Flow Matching (FM)**: An introduction to the simulation-free approach for training Continuous Normalizing Flows.

## [Part 2: Flow Matching: A Deep Dive](./part02_FM)

Here, we explore the theoretical foundations and mathematical principles behind Flow Matching.

- **Normalizing Flows**: The foundational concept of transforming simple distributions into complex ones.
- **Continuous Normalizing Flows (CNFs)**: Generalizing discrete flows to continuous-time transformations using ordinary differential equations (ODEs).
- **Flow Matching (FM)**: A paradigm for training CNFs by directly regressing vector fields, avoiding costly simulations.
- **Conditional Flow Matching (CFM)**: A practical, tractable training method using Optimal Transport (OT) paths.

## [Part 3: F5-TTS: A Case Study in Flow Matching](./part03_F5TTS)

This section details F5-TTS, a cutting-edge, fully non-autoregressive Text-to-Speech model that showcases the power of Flow Matching.

- **F5-TTS Architecture**: An overview of its components, including the Diffusion Transformer (DiT) backbone.
- **Guidance Mechanisms**:
    - **Classifier Guidance (CG)**: Enhancing conditional fidelity using an external classifier.
    - **Classifier-Free Guidance (CFG)**: A simpler, more effective method that avoids the need for a separate classifier.
- **Training and Sampling**: A look at the practical implementation, including the CFM loss and the "Sway" sampling technique for efficient inference.