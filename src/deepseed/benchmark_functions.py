import torch

def sphere(x):
    """Sphere function. Global minimum 0 at x = (0, ..., 0)."""
    return torch.sum(x**2, dim=-1)

def rastrigin(x, A=10):
    """Rastrigin function. Global minimum 0 at x = (0, ..., 0)."""
    n = x.shape[-1]
    x = x.float() # Ensure float type
    term1 = x**2
    term2 = - A * torch.cos(2 * torch.pi * x)
    return A * n + torch.sum(term1 + term2, dim=-1)

def rosenbrock(x):
    """Rosenbrock function (Banana function). Global minimum 0 at x = (1, ..., 1)."""
    x = x.float() # Ensure float type
    return torch.sum(100.0 * (x[..., 1:] - x[..., :-1]**2)**2 + (x[..., :-1] - 1.0)**2, dim=-1)

def ackley(x, a=20, b=0.2, c=2 * torch.pi):
    """Ackley function. Global minimum 0 at x = (0, ..., 0)."""
    n = x.shape[-1]
    x = x.float() # Ensure float type
    sum_sq_term = torch.sum(x**2, dim=-1)
    cos_term = torch.sum(torch.cos(c * x), dim=-1)
    term1 = -a * torch.exp(-b * torch.sqrt(sum_sq_term / n))
    term2 = -torch.exp(cos_term / n)
    return term1 + term2 + a + torch.exp(torch.tensor(1.0, device=x.device, dtype=x.dtype)) # Match device and dtype

def griewank(x):
    """Griewank function. Global minimum 0 at x = (0, ..., 0)."""
    n = x.shape[-1]
    x = x.float() # Ensure float type
    sum_term = torch.sum(x**2 / 4000.0, dim=-1)
    # Ensure arange tensor is on the correct device and dtype
    denominators = torch.sqrt(torch.arange(1, n + 1, device=x.device, dtype=x.dtype))
    cos_term = torch.cos(x / denominators)
    prod_term = torch.prod(cos_term, dim=-1)
    return sum_term - prod_term + 1.0

def schwefel(x):
    """Schwefel function. Global minimum 0 at x = (420.9687, ..., 420.9687)."""
    n = x.shape[-1]
    x = x.float() # Ensure float type
    term1 = 418.9829 * n
    term2 = torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)
    return term1 - term2

def levy(x):
    """Levy function. Global minimum 0 at x = (1, ..., 1)."""
    n = x.shape[-1]
    x = x.float() # Ensure float type
    # Calculate w
    w = 1.0 + (x - 1.0) / 4.0
    # Calculate terms
    term1 = torch.sin(torch.pi * w[..., 0])**2
    term3 = (w[..., -1] - 1.0)**2 * (1.0 + torch.sin(2.0 * torch.pi * w[..., -1])**2)
    # Slicing for intermediate terms
    if n > 1:
        wi = w[..., :-1] # All dimensions except the last
        sum_term = torch.sum((wi - 1.0)**2 * (1.0 + 10.0 * torch.sin(torch.pi * wi + 1.0)**2), dim=-1)
    else: # Handle 1D case
        sum_term = torch.zeros_like(term1)
    return term1 + sum_term + term3

def michalewicz(x, m=10):
    """Michalewicz function. Global minimum depends on dimension, occurs for x_i near pi. m=10 is standard."""
    n = x.shape[-1]
    x = x.float() # Ensure float type
    i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype) # Ensure i is on the same device and type
    term = torch.sin(x) * torch.sin((i * x**2) / torch.pi)**(2 * m)
    return -torch.sum(term, dim=-1)


# Dictionary mapping function names to (function, bounds)
BENCHMARK_FUNCTIONS = {
    "Sphere": (sphere, (-5.12, 5.12)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-5.0, 10.0)), # Wider bounds often used
    "Ackley": (ackley, (-32.768, 32.768)),
    "Griewank": (griewank, (-600.0, 600.0)),
    "Schwefel": (schwefel, (-500.0, 500.0)),
    "Levy": (levy, (-10.0, 10.0)),
    "Michalewicz": (michalewicz, (0.0, torch.pi)), # Note: bounds are [0, pi]
}
