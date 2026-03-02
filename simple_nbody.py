import numpy as np
import time

class SimpleNBody:
    """
    A pure Python, NumPy-vectorized N-Body integrator for gravity.
    Best for N < 5000.
    """
    def __init__(self, pos, vel, mass, softening=0.1, G=1.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.mass = np.array(mass, dtype=float)
        self.softening = softening
        self.G = G
        self.N = len(mass)
        self.acc = self._compute_acc() # Initial acceleration

    def _compute_acc(self):
        """Vectorized gravity calculation (O(N^2))"""
        # 1. Distance vectors (N x N x 3)
        # Using broadcasting: (N,1,3) - (1,N,3)
        dx = self.pos[:, 0:1] - self.pos[0:1, 0]
        dy = self.pos[:, 1:2] - self.pos[0:1, 1]
        dz = self.pos[:, 2:3] - self.pos[0:1, 2]

        # 2. Distance squared + Softening
        r2 = dx**2 + dy**2 + dz**2 + self.softening**2
        
        # 3. Inverse cube
        inv_r3 = r2**(-1.5)
        
        # 4. Multiply by Mass (G * m_j / r^3)
        acc_factor = self.G * inv_r3 * self.mass[None, :]
        np.fill_diagonal(acc_factor, 0.0) # No self-force

        # 5. Sum forces
        ax = -np.sum(acc_factor * dx, axis=1)
        ay = -np.sum(acc_factor * dy, axis=1)
        az = -np.sum(acc_factor * dz, axis=1)
        
        return np.column_stack((ax, ay, az))

    def step(self, dt):
        """Leapfrog Integration Step"""
        # Kick (v + 0.5*a*dt)
        self.vel += 0.5 * self.acc * dt
        
        # Drift (x + v*dt)
        self.pos += self.vel * dt
        
        # Update Acceleration
        self.acc = self._compute_acc()
        
        # Kick (v + 0.5*a*dt)
        self.vel += 0.5 * self.acc * dt
        
        return self.pos, self.vel

# ============================================================
# EXAMPLE USAGE (For the User)
# ============================================================
if __name__ == "__main__":
    # Create a random cluster
    N = 1000
    pos = np.random.randn(N, 3)
    vel = np.zeros((N, 3))
    mass = np.ones(N) / N
    
    sim = SimpleNBody(pos, vel, mass)
    
    print(f"Running simulation with {N} particles...")
    t0 = time.time()
    
    for i in range(100):
        sim.step(dt=0.01)
        if i % 10 == 0:
            print(f"Step {i}")
            
    print(f"Done in {time.time()-t0:.2f} seconds.")
