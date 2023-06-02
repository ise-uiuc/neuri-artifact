# NeuRI: Diversifying DNN Generation via Inductive Rule Inference

<p align="center">
    <a href="https://arxiv.org/abs/2302.02261"><img src="https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg">
    <a href="https://github.com/ise-uiuc/neuri-artifact/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

Welcome to the artifact repository of the NeuRI paper which is accepted by ESEC/FSE 2023.

## Bugs (RQ3)

> **Note** Annotations
> * **Status**: ✅ fixed; 🚨 high-priority bug; 🔵 explicitly confirmed by developers; ❌ won't fix;
> * **Symptom**: 💥 Crash or exception; 🧮 Result inconsistency (silent semantic bug); 🧴 Sanitizers;

### PyTorch

1. ✅💥🚨 [SIGIOT when running model with conv2d and avgpool2d after `optimize_for_inference` · Issue #86535 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86535)
2. ✅🧮 [`optimize_for_inference` leads to wrong results for model with conv2d, max and clip · Issue #86556 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86556)
3. ✅💥🚨 [RuntimeError: could not construct a memory descriptor using a format tag · Issue #86664 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86664)
4. ✅💥 [[NNPack] Runtime error with padded `Conv1d` and `&gt;=16` batch size · Issue #90142 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90142)
5. ✅💥 [stable `torch.sort` crash with expanded tensor · Issue #91420 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91420)
6. ✅💥 [[Crash] `torch.searchsorted` with out-of-bound sorter · Issue #91606 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91606)
7. ✅🧮 [`index_select` with scalar input and 0-dimed vector leads to undeterministic output · Issue #94340 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94340)
8. ✅🧮 [`index_select` with scalar input and 0-dimed vector leads to undeterministic output · Issue #94340 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94340)
9. ✅💥 [`torch.compile` failed on `torch.add` with a constant python number · Issue #92324 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92324)
10. ✅💥 [`torch.compile` generates wrong profiling program for `randn_like` · Issue #92368 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92368)
11. ✅💥 [`torch.compile` generates wrong profiling program for function having `transpose` and `lerp` · Issue #93229 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93229)
12. ✅💥 [`torch.compile` triggers assertion error when explicitly provide `out=None` · Issue #92814 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92814)
13. ✅💥 [INTERNAL ASSERT FAILED in `torch.compile` when the input tensor of `torch.clamp` has `requires_grad=True` · Issue #93225 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93225)
14. ✅💥 [`torch.compile` failed to run in-place operation `unsqueeze_(0)` · Issue #93259 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93259)
15. ✅🧮 [`stack` + inplace operator produce wrong results in `torch.compile` · Issue #93283 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93283)
16. ✅🧮 [[pt2] compiled model with cat and expand gives wrong results · Issue #93357 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93357)
17. ✅🧮🚨 [[pt2] compiled function with cat and mul gives wrong results · Issue #93365 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93365)
18. ✅🧮 [[pt2] cannot compile model with linear layer when the input has rank 1 · Issue #93372 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93372)
19. ✅💥 [`softmax` + `transpose` + `div_` triggers assertion fail in compile mode · Issue #93371 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93371)
20. ✅🧮🚨 [[pt2] `torch.where` gives wrong results with `torch.compile` · Issue #93374 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93374)
21. ✅💥 [`torch.rsub` with `alpha=xxx` triggers assertion fail in compile mode · Issue #93376 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93376)
22. ✅🧮 [[pt2] compile gives wrong result for function having `expand` and `div_` · Issue #93377 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93377)
23. ✅💥 [[pt2] `torch._inductor.exc.CppCompileError: C++ compile error` when compiling `neg` and `max` · Issue #93380 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93380)
24. ✅💥 [[pt2] exception when compiling `max_pool2d_with_indices` · Issue #93384 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93384)
25. ✅💥 [[pt2] cannot compile function having `gt`, `expand` and `add_` · Issue #93386 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93386)
26. ✅💥🚨 [`torch.compile` trigger assertion error when executing `histogramdd` · Issue #93274 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93274)
27. ✅🧮 [[pt2] `torch.compile` produces wrong results for `masked_fill` · Issue #93823 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93823)
28. ✅🧮 [[pt2] `torch.compile` produces wrong results for function with `reciprocal_` · Issue #93824 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93824)
29. ✅🧮 [[pt2] `torch.compile` produces wrong results for function with `neg` on `uint8` tensor · Issue #93829 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93829)
30. ✅💥 [`log_softmax` + `pad` triggers assertion fail in compile mode · Issue #93819 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93819)
31. ✅💥 [[pt2] Cannot compile model with `neg` and `linear` · Issue #93836 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93836)
32. ✅🧮 [`pad` + `gt` produce wrong results in compile mode · Issue #93351 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93351)
33. ✅💥 [[pt2] (`interpolate` with `mode=nearest`) + `kthvalue` triggers assertion fail in compile mode · Issue #93849 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93849)
34. ✅💥 [[pt2] `torch._inductor.exc.CppCompileError: C++ compile error` when compiling `argmax` and `min` · Issue #94055 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94055)
35. ✅💥 [`Tensor.select` + `add_` triggers C++ Compile Error · Issue #94960 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94960)
36. ✅💥 [`torch.compile` fails when using `torch.sub` with python constant · Issue #95181 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95181)
37. ✅💥 [`Tensor.copy_` + `moveaxis` Trigger Exception in Compile Mode · Issue #95262 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95262)
38. ✅🧮 [`torch.ge` produces wrong results in compile mode when given int tensors · Issue #95695 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95695)
39. ✅💥 [[pt2] `bitwise_and` + `clamp_max` Triggers Compilation Error · Issue #97968 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/97968)
40. ✅🧮 [[pt2] `add` + `unfold` + `abs_` produces wrong results · Issue #98143 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/98143)
41. ✅🧮 [[pt2] `pow` + `cos` produces wrong result · Issue #98149 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/98149)
42. ✅💥 [`torch._C._nn.fractional_max_pool3d` Trigger Segmentation Fault · Issue #89648 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89648)
43. ✅💥🚨 [`torch.nn.functional.embedding_bag` Trigger &quot;IOT instruction&quot; Failure · Issue #89677 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89677)
44. ✅🧴 [`torch.Tensor.index_select` Trigger heap-buffer-overflow with AddressSanitizer · Issue #88940 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88940)
45. ✅🧴 [`nn.utils.rnn.pack_sequence` Trigger heap-buffer-overflow with AddressSanitizer · Issue #88334 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88334)
46. ✅🚨🧴 [`MultiMarginLoss` Trigger out-of-bound Read under Compute Sanitizer · Issue #88724 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88724)
47. ✅🧴 [`nn.functional.max_unpool3d` Trigger heap-buffer-overflow with AddressSanitizer · Issue #88032 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88032)
48. ✅🧴 [`torch.nn.functional.interpolate` Trigger heap-buffer-overflow with AddressSanitizer  · Issue #88939 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88939)
49. ✅🧴 [`torch.fft.hfft` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88985 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88985)
50. ✅🧴 [`torch.nn.functional.interpolate` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88951 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88951)
51. 🔵💥 [[JIT] Zero-channel conv2d cannot be applied with `optimize_for_inference` · Issue #91396 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91396)
52. 🔵💥 [[JIT] Applying `conv2d` over Constants Leads to Exception · Issue #92740 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92740)
53. 🔵💥🚨 [`torch.compile` failed on `torch.bitwise_xor` with a constant python number · Issue #93224 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93224)
54. 🔵🧮 [`dstack` + `reciprocal` produce wrong result in compile mode · Issue #93078 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93078)
55. 🔵💥 [`min` reduction on float16 tensor failed on certain shapes · Issue #93249 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93249)
56. 🔵🧮 [`torch.compile` produce wrong result in `interpolate` when `mode=bilinear` · Issue #93262 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93262)
57. 🔵💥 [`argmin` + `view` Trigger Exception in compile mode · Issue #95370 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95370)
58. 🔵🧮 [`torch.fmod` produces inconsistent results in eager and compile mode · Issue #97333 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/97333)
59. 🔵💥 [[CPU Inductor] Compile error when passing float16 tensors to `vector_norm` + `remainder` · Issue #97758 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/97758)
60. 🔵💥 [[pt2] `movedim` + `add_` + `cat` triggers exception · Issue #98122 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/98122)
61. 🔵💥 [`torch.Tensor.flatten` Trigger Segmentation Fault when trying to provide and output named dim · Issue #89718 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89718)
62. 🔵🧴 [`nn.functional.embedding_bag` Trigger out-of-bound Read under Compute Sanitizer · Issue #88563 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88563)
63. 🔵🧴 [`torch.nn.CTCLoss` Trigger heap-buffer-overflow under AddressSanitizer · Issue #88047 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88047)
64. 🔵🧴 [`torch.nn.ReplicationPad2D` Report &quot;invalid configuration argument&quot; Error under Compute Sanitizer · Issue #89254 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89254)
65. 🔵🧴 [`torch.nn.LayerNorm` Abort with &quot;invalid device ordinal&quot; Error · Issue #89218 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89218)
66. 🔵🧴 [`torch.svd_lowrank` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88942 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88942)
67. 🔵🧴 [`torch.linalg.lstsq` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88941 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88941)
68. 💥 [Adding a linear layer leads to failure of `optimize_for_mobile` · Issue #86667 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86667)
69. 💥 [[JIT] INTERNAL ASSERT FAILED when dispatching for `torch.Tensor.view` · Issue #90365 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90365)
70. 💥 [[JIT] INTERNAL ASSERT FAILED `torch.add` with boolean primitive constant · Issue #90367 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90367)
71. 💥 [[JIT] INTERNAL ASSERT FAILED `torch.mul` with boolean primitive constant · Issue #90366 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90366)
72. 💥 [[JIT] Wrong type inference leads to misleading error message · Issue #90369 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90369)
73. 💥 [[JIT] INTERNAL ASSERT FAILED when `Conv2d` and `clamp` used together · Issue #92563 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92563)
74. 💥 [[JIT] Inconsistency  in tensor shape between eager mode and JIT · Issue #92548 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92548)
75. 💥 [[JIT][TracingCheckError] inplace ops incompatible with `contiguous(.., channels_last)` · Issue #92558 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92558)
76. ❌💥 [[JIT] `Linear` + `BatchNorm2d` Trigger Inconsistency between Eager Mode and JIT · Issue #92674 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92674)
77. 💥 [[JIT] Consecutive use of `addmm` Leads to Exception · Issue #92742 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92742)
78. ❌💥 [[Crash][MKL] `torch.linalg.eigvals` crash with NaN · Issue #93124 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93124)
79. 🧴 [`torch.nn.CTCLoss` Trigger out-of-bound Read under Compute Sanitizer · Issue #89208 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89208)
80. 🧴 [`torch.nn.functional.embedding_bag` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88950 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88950)
81. 🧴 [`torch.set_rng_state` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88949 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88949)
82. 🧴 [`torch.Tensor.msort` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88947 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88947)
83. 🧴 [`torch.linalg.eigvals` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88945 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88945)
84. 🧴 [`torch.topk` Trigger RuntimError under UndefinedBehaviorSanitizer · Issue #88944 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88944)
85. 🧴 [`torch.vander` Trigger RuntimeError with UndefinedBehaviorSanitizer · Issue #88943 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88943)

### TensorFlow

1. 🔵💥 [Inconsistant behavior of Conv2D between eager mode and tracing · Issue #57664 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57664)
2. 🔵💥 [TFLite fails to run a model with a dense layer following an Add operator · Issue #57697 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57697)
3. 🔵💥 [TFLite throws an error with certain tensor value · Issue #57708 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57708)
4. 🔵🧮 [TFLite&#39;s max operator has wrong broadcasting behavior · Issue #57759 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57759)
5. 🔵💥 [Issues · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/58035 )
6. 🔵🧮 [pow operation gives valid output even the input is invalid · Issue #57757 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57757)
7. 🔵🧮 [TFLite produce wrong results when add follows a leakyrelu · Issue #57818 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57818)
8. 🔵💥 [TFLite runner crashes with XOR and squeeze in the model · Issue #57882 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57882)
9. 🔵💥 [ Conv2D with XLA jit_compile=True fails to run · Issue #57748 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57748)
10. 🔵🧮 [log operator outputs wrong results with XLA compilation · Issue #57744 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57744)
11. 🔵🧮 [pow operator output nan for valid inputs · Issue #57747 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57747)
12. 🔵🧮 [LRN operator outputs wrong results with `jit_compile=True` · Issue #57746 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57746)
13. 🔵💥 [Conv2D layer fails to run with XLA on CUDA · Issue #57838 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57838)
14. 🔵🧴 [`tf.raw_ops.SegmentMax` Behaves Differently Under CPU and GPU · Issue #58469 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/58469)
15. ❌🧮 [Inconsistent behavior of TF eager and XLA in int64 casting · Issue #57883 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57883)

> **Note**
> NeuRI or NNSmith is able to find a lot more TensorFlow bugs in addition to these above.
> Because we want to report bugs more [responsively](https://blog.regehr.org/archives/2037), we use a buffer size of 15 reports until some of them are fixed.
> That being said, the 15 reported bugs are not fixed (though confirmed) by TensorFlow developers to date and we thus discontinue finding bugs over TensorFlow.

## Evaluating Coverage (RQ2)

> **Warning** TBD

## Evaluating Rule Inference (RQ3)

> **Warning** Experiment dependency.
> You need to first finish the last section (RQ2) to continue this section.

## Learning More

- Pre-print: [![](https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg)](https://arxiv.org/abs/2302.02261)
- NeuRI is being merged into [NNSmith](https://github.com/ise-uiuc/nnsmith)
