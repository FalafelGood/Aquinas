Aquinas (**A** **qu**antum **in**terferometer **as**sembler) is a rudimentary software package for converting linear interferometers into digital quantum circuits. Given a maximum photon number $n$ and an $m \times m$ unitary matrix $U$ (that describes how the creation operators of an $m$ mode linear interferometer are transformed) our `direct_decomposition()` method returns a quantum circuit with around $m \log_2(n)$ qubits that can simulate the interferometer for any number state containing up to (and including) $n$ photons.

We work in the second quantization picture to encode each of the $m$ interferometer modes with an $n+1$ dimensional quantum register containing between $0$ and $n$ indistinguishable photons. For details about our algorithm, please consult our whitepaper (soon to be linked here).

It is recommended that you build this package in a *virtual environment* (conda or venv for example).
When ready to install, simply run:

`pip install Aquinas`

Please constact me if there are any issues. I can be reached at leoneht0 at gmail dot com.
