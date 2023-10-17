import jax
import jax.numpy as jnp
import itertools
import numpy as np
from jax import jit, vmap
#use float 64 tensors
from jax.config import config
config.update("jax_enable_x64", True)

def setup_fourier(n, b, T):
    B = 2**b
    # function definition

    @jit
    def convert_vec_to_binary(v):
        dim = v.shape[0]
        return jnp.dot(2 ** jnp.arange(dim-1, -1, -1, dtype=jnp.int32), v)

    hash_to_index = jit(vmap(convert_vec_to_binary, in_axes=(1)))
    #hash_to_index = vmap(convert_vec_to_binary, in_axes=(1))

    @jit
    def run(ref_wht, shifted_wht, freqs, amps, seed):
        """
        ref_wht is 2^b array. Coressponding to zero shift
        shifted_wht is n * 2^b array. Coressponding to e_i shifts
        freqs is n * B array consisiting of current estimate of frequencies
        amps is 2B array consisting of current estimates of amplitudes
        """
        # hashing matrix setup
        key = jax.random.PRNGKey(seed)
        hashing_matrix = jax.random.randint(key=key, shape=(n, b), minval=0, maxval=2)
        # Hash frequenices to buckets and peel
        hashed_freqs = (hashing_matrix.T @ freqs) % 2
        index = hash_to_index(hashed_freqs)
        ref_wht_peeled = ref_wht.at[index].add(-amps)
        signed_amps = jnp.where(freqs == 0, 1, -1) * amps
        shifted_wht_peeled = shifted_wht.at[:, index].add(-signed_amps)
        # recover requencies, n * B array
        recovered_freqs = jnp.where(jnp.sign(ref_wht_peeled) == jnp.sign(shifted_wht_peeled), 0, 1)
        return recovered_freqs, ref_wht_peeled

    # run it once to get it compiled
    run(ref_wht=jnp.zeros(B), shifted_wht=jnp.zeros((n, B)), freqs=jnp.zeros((n, B), dtype=jnp.int32), amps=jnp.zeros(B), seed=0)
    @jit
    def sparse_wht(ref_whts, shifted_whts):
        freqs = jnp.zeros((n, B), dtype=jnp.int32)
        amps = jnp.zeros(B)
        for i in range(T):
            recovered_freqs, recovered_amps = run(ref_whts[i], shifted_whts[i], freqs, amps, i)
            all_freqs=  jnp.hstack([freqs, recovered_freqs])
            all_amps = jnp.concatenate([amps, recovered_amps])
            unique_freqs, index = jnp.unique(all_freqs, axis=1, return_inverse=True, size=2*B, fill_value=0)
            unique_amps = jnp.zeros(2*B)
            unique_amps = unique_amps.at[index].add(all_amps)
            index = jnp.argsort(jnp.abs(unique_amps))[B:2*B]

            freqs = freqs.at[:, :].set(unique_freqs[:,index])
            amps = amps.at[:].set(unique_amps[index])

        return freqs, amps


    # run it once to get it compiled
    sparse_wht(ref_whts=jnp.zeros((T, B)), shifted_whts=jnp.zeros((T, n, B)))
    return sparse_wht


def get_time_samples(n, b, T):
    t_list = jnp.array(list(itertools.product([0, 1], repeat=b))).T
    ret_value = []
    for i in range(T):
        key = jax.random.PRNGKey(i)
        hashing_matrix = jax.random.randint(key=key, shape=(n, b), minval=0, maxval=2)
        time_samples = (hashing_matrix @ t_list) % 2
        ret_value.append(time_samples)
    # T * n * 2^b array have to add the shifts yourself
    return np.array(ret_value)