import networkx as nx
from typing import List, Dict, Any
import numpy as np
from neighbor_allocation import define_applications
import time

def run_pso_allocation(graph: nx.Graph, app_count: int, config: Dict[str, Any], external_cloud: int) -> Dict[str, Any]:
    # Start the PSO allocation process and record the total execution time
    start_total = time.time()
    print(f"Starting run_pso_allocation at: {(time.time() - start_total):.6f} sec")
    
    # Identify nodes in the graph and classify them as regular or external cloud nodes
    start_nodes = time.time()
    fog_nodes = [n for n in graph.nodes if graph.nodes[n].get('node_type') == 'regular']
    all_nodes = fog_nodes + [external_cloud]
    num_nodes = len(all_nodes)
    if not all_nodes:
        raise ValueError("No nodes available for allocation.")
    
    # Define applications based on the provided configuration
    start_apps = time.time()
    apps = define_applications(app_count, config)
    
    # Initialize PSO parameters and variables
    num_particles = max(300, app_count // 10)
    max_iterations = 20
    w = 0.5  # Inertia weight
    c1 = 1.0  # Cognitive coefficient
    c2 = 3.0  # Social coefficient

    total_cpu = sum(graph.nodes[node]['cpu'] for node in graph.nodes)
    avg_cpu_per_app = total_cpu / app_count
    concurrency_limit = max(500, int(avg_cpu_per_app / 2))

    results = {
        'allocated_count': 0, 'fog_node_allocations': 0, 'cloud_allocations': 0,
        'utilized_fog_nodes': set(), 'total_CPU': 0.0, 'total_RAM': 0.0,
        'total_storage': 0.0, 'total_Bandwidth': 0.0, 'total_Latency': 0.0,
        'total_Makespan': 0.0, 'total_Workload': 0.0, 'total_Energy': 0.0,
        'total_Cost': 0.0, 'allocation_status': [], 'iteration_scores': []
    }

    # Prepare data structures to track node resources and application allocations
    node_resources = {node: {'cpu_remaining': graph.nodes[node]['cpu'],
                            'ram_remaining': graph.nodes[node]['ram'],
                            'storage_remaining': graph.nodes[node]['storage']} for node in graph.nodes}
    node_makespans = {node: 0.0 for node in graph.nodes}
    node_app_counts = {node: 0 for node in graph.nodes}

    # Initialize particles, velocities, and fitness values for the PSO algorithm
    start_pso = time.time()
    particles = []
    velocities = []
    pbest_positions = []
    pbest_fitness = []
    gbest_position = None
    gbest_fitness = float('inf')

    for _ in range(num_particles):
        position = np.random.randint(0, num_nodes, size=app_count).tolist()
        velocity = np.random.uniform(-1, 1, size=app_count).tolist()
        particles.append(position)
        velocities.append(velocity)
        pbest_positions.append(position.copy())
        pbest_fitness.append(float('inf'))

    # Define the fitness evaluation function to calculate the cost of a particle's position
    def evaluate_fitness(position: List[int]) -> float:
        # Evaluate the fitness of a particle's position based on resource usage, latency, and cost
        temp_resources = {node: {'cpu_remaining': graph.nodes[node]['cpu'],
                                'ram_remaining': graph.nodes[node]['ram'],
                                'storage_remaining': graph.nodes[node]['storage']} for node in graph.nodes}
        temp_makespans = {node: 0.0 for node in graph.nodes}
        temp_app_counts = {node: 0 for node in graph.nodes}
        allocated_apps = 0
        total_cost = 0.0
        total_latency = 0.0
        total_energy = 0.0
        cloud_penalty = 0

        for app_idx, node_idx in enumerate(position):
            app = apps[app_idx]
            target_node = all_nodes[node_idx]
            if (temp_resources[target_node]['cpu_remaining'] >= app['CPU'] and
                temp_resources[target_node]['ram_remaining'] >= app['RAM'] and
                temp_resources[target_node]['storage_remaining'] >= app['Storage']):
                allocated_apps += 1
                temp_resources[target_node]['cpu_remaining'] -= app['CPU']
                temp_resources[target_node]['ram_remaining'] -= app['RAM']
                temp_resources[target_node]['storage_remaining'] -= app['Storage']
                temp_app_counts[target_node] += 1
                if target_node == external_cloud:
                    cloud_penalty += 1
                try:
                    path = nx.shortest_path(graph, source=target_node, target=external_cloud, weight='propagation_delay')
                    bandwidth = min(graph.edges[edge]['bandwidth'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_bandwidth']
                    latency = sum(graph.edges[edge]['propagation_delay'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_propagation_delay']
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    bandwidth = config['EXTERNAL_CLOUD']['c_bandwidth']
                    latency = config['EXTERNAL_CLOUD']['c_propagation_delay']
                total_latency += latency
                app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
                concurrent_slots = max(1, graph.nodes[target_node]['cpu'] // concurrency_limit)
                if target_node == external_cloud:
                    concurrent_slots = min(concurrent_slots, 500)
                if temp_app_counts[target_node] <= concurrent_slots:
                    temp_makespans[target_node] = max(temp_makespans[target_node], app_makespan)
                else:
                    temp_makespans[target_node] += app_makespan
                energy = (config['ENERGY_PARAMETERS']['p_idle'] * app['Runtime'] +
                          config['ENERGY_PARAMETERS']['p_pro'] * app['CPU'] / graph.nodes[target_node]['cpu'] * app['Runtime'] +
                          config['ENERGY_PARAMETERS']['p_trans'] * app['Msg_Size'] / bandwidth)
                cost = (energy * config['COST_PARAMETERS']['unit_cost_energy'] +
                        app['CPU'] * config['COST_PARAMETERS']['unit_cost_cpu'] * app['Runtime'] +
                        app['RAM'] * config['COST_PARAMETERS']['unit_cost_ram'] * app['Runtime'] +
                        app['Storage'] * config['COST_PARAMETERS']['unit_cost_storage'] * app['Runtime'])
                total_energy += energy
                total_cost += cost
        makespan = max(temp_makespans.values()) if temp_makespans else 0.0
        unallocated_penalty = (app_count - allocated_apps) * 10000
        fitness = unallocated_penalty + makespan * 5000 + total_cost * 0.5 + total_latency * 10 + total_energy * 0.5 + cloud_penalty * 1000
        return fitness

    # Perform the PSO iterations to optimize the allocation of applications
    start_iter = time.time()
    for iteration in range(max_iterations):
        # Update personal best and global best positions for each particle
        for i in range(num_particles):
            fitness = evaluate_fitness(particles[i])
            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest_positions[i] = particles[i].copy()
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = particles[i].copy()

        # Log the best fitness score for the current iteration
        results['iteration_scores'].append(gbest_fitness)
        print(f"Iteration {iteration + 1}/{max_iterations}, Best Score: {gbest_fitness:.3f}")
        if iteration == 0:
            print(f"Time to first iteration: {(time.time() - start_iter):.3f} sec")

        # Update particle velocities and positions based on PSO equations
        for i in range(num_particles):
            for j in range(app_count):
                r1, r2 = np.random.random(), np.random.random()
                velocities[i][j] = (w * velocities[i][j] +
                                    c1 * r1 * (pbest_positions[i][j] - particles[i][j]) +
                                    c2 * r2 * (gbest_position[j] - particles[i][j]))
                new_pos = particles[i][j] + int(np.tanh(velocities[i][j]) * num_nodes)
                particles[i][j] = max(0, min(num_nodes - 1, new_pos))

    # Allocate applications based on the best particle's position
    allocated_apps = set()
    for app_idx, node_idx in enumerate(gbest_position):
        app = apps[app_idx]
        target_node = all_nodes[node_idx]
        allocated = False
        # Check if the application can be allocated to the target node
        if (node_resources[target_node]['cpu_remaining'] >= app['CPU'] and
            node_resources[target_node]['ram_remaining'] >= app['RAM'] and
            node_resources[target_node]['storage_remaining'] >= app['Storage']):
            allocated = True
            allocated_apps.add(app_idx)
            # Update node resources and allocation results
            node_resources[target_node]['cpu_remaining'] -= app['CPU']
            node_resources[target_node]['ram_remaining'] -= app['RAM']
            node_resources[target_node]['storage_remaining'] -= app['Storage']
            node_app_counts[target_node] += 1
            results['allocated_count'] += 1
            if graph.nodes[target_node].get('node_type') == 'regular':
                results['fog_node_allocations'] += 1
                results['utilized_fog_nodes'].add(target_node)
            elif graph.nodes[target_node].get('node_type') == 'cloud':
                results['cloud_allocations'] += 1
            results['total_CPU'] += app['CPU']
            results['total_RAM'] += app['RAM']
            results['total_storage'] += app['Storage']
            try:
                path = nx.shortest_path(graph, source=target_node, target=external_cloud, weight='propagation_delay')
                bandwidth = min(graph.edges[edge]['bandwidth'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_bandwidth']
                latency = sum(graph.edges[edge]['propagation_delay'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_propagation_delay']
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                bandwidth = config['EXTERNAL_CLOUD']['c_bandwidth']
                latency = config['EXTERNAL_CLOUD']['c_propagation_delay']
            results['total_Bandwidth'] += bandwidth
            results['total_Latency'] += latency
            app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
            concurrent_slots = max(1, graph.nodes[target_node]['cpu'] // concurrency_limit)
            if target_node == external_cloud:
                concurrent_slots = min(concurrent_slots, 500)
            if node_app_counts[target_node] <= concurrent_slots:
                node_makespans[target_node] = max(node_makespans[target_node], app_makespan)
            else:
                node_makespans[target_node] += app_makespan
            workload = app['CPU'] * 0.3 + app['RAM'] * 0.2 + app['Storage'] * 0.2 + app['Runtime'] * 0.3
            energy = (config['ENERGY_PARAMETERS']['p_idle'] * app['Runtime'] +
                      config['ENERGY_PARAMETERS']['p_pro'] * app['CPU'] / graph.nodes[target_node]['cpu'] * app['Runtime'] +
                      config['ENERGY_PARAMETERS']['p_trans'] * app['Msg_Size'] / bandwidth)
            cost = (energy * config['COST_PARAMETERS']['unit_cost_energy'] +
                    app['CPU'] * config['COST_PARAMETERS']['unit_cost_cpu'] * app['Runtime'] +
                    app['RAM'] * config['COST_PARAMETERS']['unit_cost_ram'] * app['Runtime'] +
                    app['Storage'] * config['COST_PARAMETERS']['unit_cost_storage'] * app['Runtime'])
            results['total_Workload'] += workload
            results['total_Energy'] += energy
            results['total_Cost'] += cost
        results['allocation_status'].append({
            'app_id': app_idx,
            'node': target_node if allocated else None,
            'allocated': allocated,
            'requirements': app
        })

    # Handle unallocated applications by finding the best node for each
    unallocated = [(app_idx, apps[app_idx]) for app_idx in range(app_count) if app_idx not in allocated_apps]
    for app_idx, app in unallocated:
        allocated = False
        best_node = None
        min_makespan_increase = float('inf')
        for node in all_nodes:
            # Check if the application can be allocated to the current node
            if (node_resources[node]['cpu_remaining'] >= app['CPU'] and
                node_resources[node]['ram_remaining'] >= app['RAM'] and
                node_resources[node]['storage_remaining'] >= app['Storage']):
                # Calculate the makespan increase for allocating the application to this node
                try:
                    path = nx.shortest_path(graph, source=node, target=external_cloud, weight='propagation_delay')
                    bandwidth = min(graph.edges[edge]['bandwidth'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_bandwidth']
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    bandwidth = config['EXTERNAL_CLOUD']['c_bandwidth']
                app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
                concurrent_slots = max(1, graph.nodes[node]['cpu'] // concurrency_limit)
                if node == external_cloud:
                    concurrent_slots = min(concurrent_slots, 500)
                current_makespan = node_makespans[node]
                new_makespan = max(current_makespan, app_makespan) if node_app_counts[node] <= concurrent_slots else current_makespan + app_makespan
                makespan_increase = new_makespan - current_makespan
                if makespan_increase < min_makespan_increase:
                    min_makespan_increase = makespan_increase
                    best_node = node
                    allocated = True
        if allocated:
            # Update node resources and allocation results for the best node
            target_node = best_node
            node_resources[target_node]['cpu_remaining'] -= app['CPU']
            node_resources[target_node]['ram_remaining'] -= app['RAM']
            node_resources[target_node]['storage_remaining'] -= app['Storage']
            node_app_counts[target_node] += 1
            results['allocated_count'] += 1
            if graph.nodes[target_node].get('node_type') == 'regular':
                results['fog_node_allocations'] += 1
                results['utilized_fog_nodes'].add(target_node)
            elif graph.nodes[target_node].get('node_type') == 'cloud':
                results['cloud_allocations'] += 1
            results['total_CPU'] += app['CPU']
            results['total_RAM'] += app['RAM']
            results['total_storage'] += app['Storage']
            try:
                path = nx.shortest_path(graph, source=target_node, target=external_cloud, weight='propagation_delay')
                bandwidth = min(graph.edges[edge]['bandwidth'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_bandwidth']
                latency = sum(graph.edges[edge]['propagation_delay'] for edge in zip(path[:-1], path[1:])) if len(path) > 1 else config['EXTERNAL_CLOUD']['c_propagation_delay']
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                bandwidth = config['EXTERNAL_CLOUD']['c_bandwidth']
                latency = config['EXTERNAL_CLOUD']['c_propagation_delay']
            results['total_Bandwidth'] += bandwidth
            results['total_Latency'] += latency
            app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
            concurrent_slots = max(1, graph.nodes[target_node]['cpu'] // concurrency_limit)
            if target_node == external_cloud:
                concurrent_slots = min(concurrent_slots, 500)
            if node_app_counts[target_node] <= concurrent_slots:
                node_makespans[target_node] = max(node_makespans[target_node], app_makespan)
            else:
                node_makespans[target_node] += app_makespan
            workload = app['CPU'] * 0.3 + app['RAM'] * 0.2 + app['Storage'] * 0.2 + app['Runtime'] * 0.3
            energy = (config['ENERGY_PARAMETERS']['p_idle'] * app['Runtime'] +
                      config['ENERGY_PARAMETERS']['p_pro'] * app['CPU'] / graph.nodes[target_node]['cpu'] * app['Runtime'] +
                      config['ENERGY_PARAMETERS']['p_trans'] * app['Msg_Size'] / bandwidth)
            cost = (energy * config['COST_PARAMETERS']['unit_cost_energy'] +
                    app['CPU'] * config['COST_PARAMETERS']['unit_cost_cpu'] * app['Runtime'] +
                    app['RAM'] * config['COST_PARAMETERS']['unit_cost_ram'] * app['Runtime'] +
                    app['Storage'] * config['COST_PARAMETERS']['unit_cost_storage'] * app['Runtime'])
            results['total_Workload'] += workload
            results['total_Energy'] += energy
            results['total_Cost'] += cost
            results['allocation_status'][app_idx] = {
                'app_id': app_idx,
                'node': target_node,
                'allocated': True,
                'requirements': app
            }

    # Finalize results and log execution details
    results['utilized_fog_nodes_count'] = len(results['utilized_fog_nodes'])
    results['total_Makespan'] = max(node_makespans.values()) if node_makespans else 0.0
    print(f"Concurrency Limit: {concurrency_limit}")
    print(f"Node Makespans: {node_makespans}")
    print(f"Apps per Node Distribution: {node_app_counts}")
    print(f"Total execution time: {time.time() - start_total} sec")
    return results

if __name__ == "__main__":
    pass