## TODO
- Replace the current MIPS hash implementation with the one specified in `A. Shrivastava and P. Li. Improved asymmetric locality sensitive hashing (alsh) for maximum inner product search (mips)`.
- Add some other hash functions:
  - Fast MinHash for Jaccard similarity hashing ([ref. 1 (Anshu's thesis)](https://www.cs.rice.edu/~as143/Doc/Anshumali_Shrivastava.pdf), [ref. 2 (ICML '17)](https://arxiv.org/pdf/1703.04664.pdf))
  - Winner-take-all hashing ([ref](http://auai.org/uai2018/proceedings/papers/321.pdf))
  - SSH ("Sketch, Shingle, and Hash") for time series ([ref](http://proceedings.mlr.press/v55/luo16.pdf))
- Add documentation.
- Integrate with popular KV stores (e.g. Redis).

### Long-term
- Add GPU support with `CuArrays.jl`.

### Old TODO items
