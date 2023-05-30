# NeuRI: Diversifying DNN Generation via Inductive Rule Inference

<p align="center">
    <a href="https://arxiv.org/abs/2302.02261"><img src="https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg">
    <a href="https://github.com/ise-uiuc/neuri-artifact/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

Welcome to the artifact repository of the NeuRI paper which is accepted by ESEC/FSE 2023.

## Bugs

> **Note:** Annotations
> * **Status**: ✅ fixed; 🚨 high-priority bug; 🔵 explicitly confirmed by developers; ❌ won't fix;
> * **Symptom**: 💥 Crash or exception; 🧮 Result inconsistency (silent semantic bug); 🧴 Sanitizers;

### PyTorch

1. ✅💥🚨 [SIGIOT when running model with conv2d and avgpool2d after `optimize_for_inference` · Issue #86535 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/86535)
2. ✅🧮 [`optimize_for_inference` leads to wrong results for model with conv2d, max and clip · Issue #86556 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/86556)
3. ✅💥🚨 [RuntimeError: could not construct a memory descriptor using a format tag · Issue #86664 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/86664)

...

### TensorFlow

...

## Learning More

- Pre-print: [![](https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg)](https://arxiv.org/abs/2302.02261)
- NeuRI is being merged into [NNSmith](https://github.com/ise-uiuc/nnsmith)
