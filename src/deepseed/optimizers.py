import torch

def pso_update(positions, velocities, pbest_positions, gbest_position, w, c1, c2, solution_dim, bounds=None):
    swarm_size = positions.shape[0]
    device = positions.device
    r1 = torch.rand(swarm_size, solution_dim, device=device)
    r2 = torch.rand(swarm_size, solution_dim, device=device)
    inertia_term = w * velocities
    cognitive_term = c1 * r1 * (pbest_positions - positions)
    social_term = c2 * r2 * (gbest_position.unsqueeze(0) - positions) # Ensure gbest is broadcastable
    new_velocities = inertia_term + cognitive_term + social_term
    new_positions = positions + new_velocities

    # Apply bounds if provided
    if bounds is not None:
        min_bound, max_bound = bounds
        # Ensure bounds are tensors on the correct device and dtype for clamp
        min_b_tensor = torch.tensor(min_bound, device=device, dtype=positions.dtype)
        max_b_tensor = torch.tensor(max_bound, device=device, dtype=positions.dtype)
        new_positions = torch.clamp(new_positions, min_b_tensor, max_b_tensor)

    return new_positions, new_velocities

def run_pso_refinement(initial_positions, fitness_function, max_pso_iterations, swarm_size, solution_dim, w_range, c1, c2, bounds=None):
    device = initial_positions.device
    positions = initial_positions.clone().detach() # Work with detached copies
    velocities = torch.zeros_like(positions, device=device)

    # Minimize the benchmark function -> Maximize -benchmark_function
    # Ensure positions are float for fitness evaluation
    fitness_vals = -fitness_function(positions.float())

    pbest_positions = positions.clone()
    pbest_fitness = fitness_vals.clone()

    # Handle potential empty swarm or fitness values
    if swarm_size > 0 and pbest_fitness.numel() > 0:
        gbest_fitness, gbest_idx = torch.max(pbest_fitness, dim=0)
        gbest_position = pbest_positions[gbest_idx].clone()
    else:
        # Handle empty swarm case (shouldn't happen in normal use, but safe)
        gbest_fitness = torch.tensor(-float('inf'), device=device, dtype=initial_positions.dtype)
        gbest_position = torch.zeros(solution_dim, device=device, dtype=initial_positions.dtype)

    w_start, w_end = w_range
    best_fitness_so_far = gbest_fitness # Track the best fitness found during PSO iterations

    for iteration in range(max_pso_iterations):
        # Check if swarm is empty before proceeding (safety check)
        if positions.shape[0] == 0:
            break

        # Linearly decreasing inertia weight
        w = w_start - (w_start - w_end) * (iteration / max_pso_iterations)

        positions, velocities = pso_update(positions, velocities, pbest_positions, gbest_position, w, c1, c2, solution_dim, bounds)

        # Evaluate current fitness
        current_fitness = -fitness_function(positions.float())

        # Update personal best
        update_pbest_mask = current_fitness > pbest_fitness
        pbest_positions[update_pbest_mask] = positions[update_pbest_mask].clone()
        pbest_fitness[update_pbest_mask] = current_fitness[update_pbest_mask].clone()

        # Update global best
        if pbest_fitness.numel() > 0: # Check if pbest_fitness is not empty
            current_best_fitness_in_swarm, current_best_idx = torch.max(pbest_fitness, dim=0)
            if current_best_fitness_in_swarm > gbest_fitness:
                gbest_fitness = current_best_fitness_in_swarm.clone()
                gbest_position = pbest_positions[current_best_idx].clone()

        # Track the absolute best fitness seen during this PSO refinement phase
        if gbest_fitness > best_fitness_so_far:
            best_fitness_so_far = gbest_fitness

    # Return the best position found and its fitness value (the best ever found in this refinement)
    # We return best_fitness_so_far as the relevant value, not just the final gbest_fitness
    return gbest_position, best_fitness_so_far.item() # Return fitness as a float

def tournament_selection(population, fitness_values, tournament_size, num_parents, device='cpu'):
    """Selects parents using tournament selection."""
    selected_parents = torch.zeros(num_parents, population.shape[1], device=device, dtype=population.dtype)
    population_size = population.shape[0]

    for i in range(num_parents):
        # Select tournament participants randomly
        participant_indices = torch.randint(0, population_size, (tournament_size,), device=device)
        # Get their fitness values (lower is better)
        participant_fitness = fitness_values[participant_indices]
        # Find the index of the best participant (minimum fitness) within the tournament
        best_participant_local_idx = torch.argmin(participant_fitness)
        # Get the index in the original population
        best_participant_global_idx = participant_indices[best_participant_local_idx]
        # Select the best participant as a parent
        selected_parents[i] = population[best_participant_global_idx]

    return selected_parents

def arithmetic_crossover(parent1, parent2, pc, bounds=None, device='cpu'):
    """Performs arithmetic crossover between two parents."""
    if torch.rand(1, device=device) > pc:
        # No crossover, return parents as offspring
        return parent1.clone(), parent2.clone()

    alpha = torch.rand(1, device=device) # Weight for crossover
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2

    # Apply bounds if provided
    if bounds is not None:
        min_b, max_b = bounds
        min_b_tensor = torch.tensor(min_b, device=device, dtype=parent1.dtype)
        max_b_tensor = torch.tensor(max_b, device=device, dtype=parent1.dtype)
        offspring1 = torch.clamp(offspring1, min_b_tensor, max_b_tensor)
        offspring2 = torch.clamp(offspring2, min_b_tensor, max_b_tensor)

    return offspring1, offspring2

def gaussian_mutation(individual, pm, mutation_strength, bounds=None, device='cpu'):
    """Applies Gaussian mutation to an individual."""
    mutated_individual = individual.clone()
    solution_dim = individual.shape[0]
    min_b, max_b = bounds if bounds else (-1e6, 1e6) # Default large bounds if none given
    min_b_tensor = torch.tensor(min_b, device=device, dtype=individual.dtype)
    max_b_tensor = torch.tensor(max_b, device=device, dtype=individual.dtype)

    # Generate mutation mask based on pm
    mutation_mask = torch.rand(solution_dim, device=device) < pm

    # Generate Gaussian noise only where mutation occurs
    noise = torch.randn(solution_dim, device=device) * mutation_strength

    # Apply mutation: add noise where mask is True
    mutated_individual[mutation_mask] += noise[mutation_mask]

    # Clamp to bounds
    mutated_individual = torch.clamp(mutated_individual, min_b_tensor, max_b_tensor)

    return mutated_individual

def run_ga_refinement(initial_population, fitness_function, max_ga_generations,
                      population_size, solution_dim, pc, pm, mutation_scale,
                      tournament_size, elitism_count, bounds=None):
    """
    Runs the Genetic Algorithm to refine a population.
    Returns the best solution (minimum objective value) found.
    """
    device = initial_population.device
    population = initial_population.clone().detach() # Work with detached copies
    min_bound, max_bound = bounds if bounds else (None, None)

    # Calculate mutation strength based on bounds range
    if bounds:
        mutation_strength = (float(max_bound) - float(min_bound)) * mutation_scale
    else:
        mutation_strength = mutation_scale # Use scale directly if no bounds

    # --- Initial Evaluation ---
    # Ensure population is float for fitness evaluation
    fitness_values = fitness_function(population.float())

    # Track overall best solution found across generations
    best_fitness_overall = torch.min(fitness_values)
    best_individual_overall = population[torch.argmin(fitness_values)].clone()

    for generation in range(max_ga_generations):
        # --- Elitism: Carry over the best individuals ---
        # Sort by fitness (ascending, lower is better)
        sorted_indices = torch.argsort(fitness_values)
        next_population = torch.zeros_like(population, device=device)
        if elitism_count > 0:
            elite_indices = sorted_indices[:elitism_count]
            next_population[:elitism_count] = population[elite_indices].clone()

        # --- Fill the rest of the population using GA operators ---
        num_offspring_needed = population_size - elitism_count
        num_parents_to_select = num_offspring_needed # Need this many parents if generating 1 offspring per pair, or double if 2

        # Handle edge case where num_offspring_needed is odd (select one extra parent if needed)
        if num_offspring_needed % 2 != 0:
            num_parents_to_select += 1

        # --- Selection ---
        parents = tournament_selection(population, fitness_values, tournament_size, num_parents_to_select, device=device)

        # --- Crossover & Mutation ---
        offspring_idx = elitism_count
        # Process parents in pairs for crossover
        for i in range(0, num_parents_to_select, 2):
             # Handle potential odd number of parents - last one doesn't crossover
            if i + 1 >= num_parents_to_select:
                 if offspring_idx < population_size: # Check if space left
                    # Mutate the single remaining parent
                    mutated_offspring = gaussian_mutation(parents[i], pm, mutation_strength, bounds, device=device)
                    next_population[offspring_idx] = mutated_offspring
                    offspring_idx += 1
                 break # Exit loop

            parent1 = parents[i]
            parent2 = parents[i+1]

            offspring1, offspring2 = arithmetic_crossover(parent1, parent2, pc, bounds, device=device)

            # Mutation
            mutated_offspring1 = gaussian_mutation(offspring1, pm, mutation_strength, bounds, device=device)
            mutated_offspring2 = gaussian_mutation(offspring2, pm, mutation_strength, bounds, device=device)

            # Add to next generation if space allows
            if offspring_idx < population_size:
                next_population[offspring_idx] = mutated_offspring1
                offspring_idx += 1
            if offspring_idx < population_size:
                next_population[offspring_idx] = mutated_offspring2
                offspring_idx += 1
            # Stop if population is full
            if offspring_idx >= population_size:
                break

        # --- Update population and evaluate fitness ---
        population = next_population
        fitness_values = fitness_function(population.float())

        # --- Update overall best ---
        current_best_fitness = torch.min(fitness_values)
        if current_best_fitness < best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_individual_overall = population[torch.argmin(fitness_values)].clone()

    # Return the best fitness value found (minimum objective value)
    return best_individual_overall, best_fitness_overall.item() # Return best individual and its fitness as float
