# LSHFunctions notation and glossary

!!! warning "Under construction"
    This section is currently being developed. If you're interested in helping write this section, feel free to [open a pull request](https://github.com/kernelmethod/LSHFunctions.jl/pulls); otherwise, please check back later.

## Terms

---

**LSH**: an acronym for locality-sensitive hashing.

---

``L^p_{\mu}(\Omega)`` **function space** ([wikipedia](https://en.wikipedia.org/wiki/Lp_space)): a set of functions[^1] whose inputs come from some set ``\Omega`` and whose outputs are either real or complex numbers. ``\mu`` is a [measure](https://en.wikipedia.org/wiki/Measure_space) and ``p`` is a positive number. The ``L^p_{\mu}(\Omega)`` [norm](https://en.wikipedia.org/wiki/Norm_(mathematics)), denoted with ``\|\cdot\|_{L^p_{\mu}}`` (where ``\Omega`` is implicit), is defined as

```math
\|f\|_{L^p_{\mu}} = \left(\int_{\Omega} \left|f(x)\right|^p \hspace{0.15cm} d\mu(x)\right)^{1/p}
```

In the case where ``p = 2``, there is also an [inner product](https://en.wikipedia.org/wiki/Inner_product_space) defined for the space:

```math
\left\langle f, g\right\rangle = \int_{\Omega} f(x)\overline{g(x)} \hspace{0.15cm} d\mu(x)
```

where ``\overline{g(x)}`` is the complex conjugate of ``g(x)``. A function in ``L^p_{\mu}(\Omega)`` must have the property that ``\|f\|_{L^p_{\mu}}`` is finite.

*Example*: ``f(x) = x^2 - 3x + 2`` is a function in ``L^2([-1,1])`` (with ``\mu`` chosen to be [Lebesgue measure](https://en.wikipedia.org/wiki/Lebesgue_measure)) because ``\|f\|_{L^2} = \sqrt{\int_{-1}^1 \left|f(x)\right|^2 \hspace{0.15cm} dx}`` is finite. However, it is *not* a function in ``L^2([-\infty,\infty])`` because the ``\|f\|_{L^2} = \sqrt{\int_{-\infty}^{\infty} \left|f(x)\right|^2 \hspace{0.15cm} dx`` is infinite.

---

**Similarity statistic**: a number that represents the similarity between two data points. Different similarity statistics have different ways of defining what "similar" means.

A similarity statistic can be interpreted in many different ways; for instance, cosine similarity is defined between -1 and 1, with *higher* values indicating *higher* similarity. Meanwhile, ``\ell^p`` distance is defined between 0 and ``+\infty``, with *higher* distances indicating *lower* similarity.

## Footnotes
[^1]: technically, equivalence classes of functions.
