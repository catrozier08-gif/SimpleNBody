# SimpleNBody

**A minimalist, vectorized N-Body gravity integrator in pure Python.**

Most N-Body codes (Gadget, Rebound) are powerful but complex to install. `SimpleNBody` is designed for education, prototyping, and small-scale simulations ($N < 5,000$). It uses NumPy vectorization to calculate $O(N^2)$ interactions efficiently without C/C++ dependencies.

## Features
*   **Pure Python:** Only requires `numpy`.
*   **Vectorized:** Uses matrix broadcasting for speed.
*   **Symplectic:** Uses the Kick-Drift-Kick (Leapfrog) integration scheme for energy conservation.
*   **Adaptive:** Handles softening to prevent singularities.

## Installation
```bash
pip install numpy
## Performance
On a standard laptop, this code can integrate:
*   $N=1,000$: ~10 steps/sec
*   $N=5,000$: ~0.5 steps/sec
