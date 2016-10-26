# Tensor network machine learning

Codes based on the paper "Supervised Learning with Quantum-Inspired Tensor Networks"
by Miles Stoudenmire and David Schwab. http://arxiv.org/abs/1605.05775

(Also see http://arxiv.org/abs/1605.03795 for a closely related approach.)


# Code Overview

`fixedL` -- optimize a matrix product state (MPS) with a label index on the central tensor, similar to what is described in the paper. This MPS parameterizes a model whose output is a vector of 10 numbers (for the case of MNIST). The output entry with the largest value is the predicted label.

`fulltest`

`single` -- optimize an MPS for a single label type, with no label index on the MPS. This MPS parameterizes a model whose output is positive for inputs of the correct type, and zero for all other inputs.

`separate_fulltest`
