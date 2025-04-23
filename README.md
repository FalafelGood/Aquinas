# About
---

Aquinas (**A** **qu**antum **in**terferometer **as**sembler) is a software package for converting linear interferometers into digital quantum circuits. Given a maximum photon number $n$ and an $m \times m$ unitary matrix $U$ (that describes how the creation operators of an $m$ mode linear interferometer are transformed) our `direct_decomposition()` method returns a quantum circuit with around $m \log_2(n)$ qubits that can simulate the interferometer for any number state containing up to (and including) $n$ photons.

We work in the second quantization picture; Each of the $m$ interferometer modes is encoded with an $n+1$ dimensional quantum register to signify that there are between $0$ and $n$ indistinguishable photons present in the mode.

For more details, please consult our whitepaper (soon to be linked here).

# Installing
---

N.B. I recommended you build this package in a *virtual environment* ([conda](https://anaconda.org/anaconda/conda) or [venv](https://docs.python.org/3/library/venv.html) for example).

When your prefered environment is configured and you're ready to install, begin by downloading the package from source. For example:

```git clone https://github.com/FalafelGood/Aquinas.git```

change into the directory, and install the package locally

```pip install .```

Note that I was having some issues before configuring the depencies. Some of them can be a bit fiddly since they don't play nice together, so please open up an issue if there are any problems.

J.M.J.
