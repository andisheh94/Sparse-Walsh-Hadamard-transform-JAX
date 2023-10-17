# Sparse-Walsh-Hadamard-transform-JAX
This is a JAX implementation of the Sparse Fourier transform from  [[1]](#1)
Look at file jax_fourier_extractor.py for a running example on how to use the transform. The only tunable parameter is the number of bins [b] (https://github.com/andisheh94/Sparse-Walsh-Hadamard-transform-JAX/blob/c5efdf43049f492f2901d9485bb612d3719f2c7d/swht_jax.py#L69). 
The larger the number the more precise and slow the algorithm will be. 


## References
<a id="1">[1]</a> 
Amrollahi, Andisheh and Zandieh, Amir and Kapralov, Michael and Krause, Andreas
Efficiently Learning Fourier Sparse Set Functions
Advances in Neural Information Processing Systems, 2019
