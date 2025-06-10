import networkx as nx
from typing import List, Dict, Any
import random

def define_applications(app_count: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Defines a list of applications with their requirements."""
    return [{
        'CPU': random.randint(config['APP_REQUIREMENTS']['a_cpu_min'], config['APP_REQUIREMENTS']['a_cpu_max']),
        'RAM': random.randint(config['APP_REQUIREMENTS']['a_ram_min'], config['APP_REQUIREMENTS']['a_ram_max']),
        'Runtime': random.randint(config['APP_REQUIREMENTS']['a_runtime_min'], config['APP_REQUIREMENTS']['a_runtime_max']),
        'Storage': random.randint(config['APP_REQUIREMENTS']['a_storage_min'], config['APP_REQUIREMENTS']['a_storage_max']),
        'Msg_Size': config['APP_REQUIREMENTS']['a_msg_size']
    } for _ in range(app_count)]

def run_neighbor_allocation(graph: nx.Graph, app_count: int, config: Dict[str, Any], external_cloud: int) -> Dict[str, Any]:
    """
    Neighbor aware Forwarding allocation method.
    Gateways forward apps to simple nodes with minimal makespan increase; if simple nodes fail, neighbors are tried;
    if all fail, the main gateway forwards to the external cloud.
    """
    # Identify node types
    simple_nodes = [n for n in graph.nodes if graph.nodes[n].get('node_type') == 'regular']
    gateway_nodes = [n for n in graph.nodes if graph.nodes[n].get('node_type') == 'gateway']
    main_gateway = max(nx.betweenness_centrality(graph).items(), key=lambda x: x[1])[0]  # Highest centrality as main gateway

    if not gateway_nodes:
        raise ValueError("No gateway nodes found in the topology. Check topology generation.")

    # Initialize results
    results = {
        'allocated_count': 0,
        'simple_node_allocations': 0,
        'cloud_allocations': 0,
        'utilized_simple_nodes': set(),
        'total_CPU': 0.0,
        'total_RAM': 0.0,
        'total_storage': 0.0,
        'total_Bandwidth': 0.0,
        'total_Latency': 0.0,
        'total_Makespan': 0.0,
        'total_Workload': 0.0,
        'total_Energy': 0.0,
        'total_Cost': 0.0,
        'allocation_status': []
    }

    # Precompute node resources and dynamic concurrency limit
    node_resources = {node: {'cpu_remaining': graph.nodes[node]['cpu'],
                            'ram_remaining': graph.nodes[node]['ram'],
                            'storage_remaining': graph.nodes[node]['storage']} for node in graph.nodes}
    node_makespans = {node: 0.0 for node in graph.nodes}
    node_app_counts = {node: 0 for node in graph.nodes}
    gateway_load = {gateway: 0 for gateway in gateway_nodes}  # Track load per gateway
    total_cpu = sum(graph.nodes[node]['cpu'] for node in graph.nodes)
    avg_cpu_per_node = total_cpu / len(graph.nodes)
    concurrency_limit = max(1000, int(avg_cpu_per_node / 2))  # Dynamic limit based on average CPU

    # Define applications
    apps = define_applications(app_count, config)

    for app_idx, app in enumerate(apps):
        allocated = False
        target_node = None

        # Step 1: Select gateway with minimal load and forward to a simple node
        selected_gateway = min(gateway_load.items(), key=lambda x: x[1])[0]  # Gateway with least apps assigned
        simple_neighbors = [n for n in graph.neighbors(selected_gateway) if n in simple_nodes]
        
        if simple_neighbors:
            min_makespan_increase = float('inf')
            best_node = None
            for neighbor in simple_neighbors:
                if (node_resources[neighbor]['cpu_remaining'] >= app['CPU'] and
                    node_resources[neighbor]['ram_remaining'] >= app['RAM'] and
                    node_resources[neighbor]['storage_remaining'] >= app['Storage']):
                    # Calculate bandwidth for this neighbor
                    bandwidth = graph.edges[(selected_gateway, neighbor)]['bandwidth']
                    app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
                    concurrent_slots = max(1, graph.nodes[neighbor]['cpu'] // concurrency_limit)
                    current_makespan = node_makespans[neighbor]
                    new_makespan = (max(current_makespan, app_makespan) if node_app_counts[neighbor] <= concurrent_slots 
                                    else current_makespan + app_makespan)
                    makespan_increase = new_makespan - current_makespan
                    if makespan_increase < min_makespan_increase:
                        min_makespan_increase = makespan_increase
                        best_node = neighbor
            if best_node:
                target_node = best_node
                allocated = True
                gateway_load[selected_gateway] += 1

        # Step 2: If not allocated, try neighbors of all gateways
        if not allocated:
            min_makespan_increase = float('inf')
            best_node = None
            best_gateway = None
            for gateway in gateway_nodes:
                neighbors = [n for n in graph.neighbors(gateway) if n in simple_nodes]
                for neighbor in neighbors:
                    if (node_resources[neighbor]['cpu_remaining'] >= app['CPU'] and
                        node_resources[neighbor]['ram_remaining'] >= app['RAM'] and
                        node_resources[neighbor]['storage_remaining'] >= app['Storage']):
                        bandwidth = graph.edges[(gateway, neighbor)]['bandwidth']
                        app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
                        concurrent_slots = max(1, graph.nodes[neighbor]['cpu'] // concurrency_limit)
                        current_makespan = node_makespans[neighbor]
                        new_makespan = (max(current_makespan, app_makespan) if node_app_counts[neighbor] <= concurrent_slots 
                                        else current_makespan + app_makespan)
                        makespan_increase = new_makespan - current_makespan
                        if makespan_increase < min_makespan_increase:
                            min_makespan_increase = makespan_increase
                            best_node = neighbor
                            best_gateway = gateway
            if best_node:
                target_node = best_node
                allocated = True
                gateway_load[best_gateway] += 1

        # Step 3: If still not allocated, main gateway forwards to external cloud
        if not allocated and external_cloud in graph.neighbors(main_gateway):
            target_node = external_cloud
            if (node_resources[external_cloud]['cpu_remaining'] >= app['CPU'] and
                node_resources[external_cloud]['ram_remaining'] >= app['RAM'] and
                node_resources[external_cloud]['storage_remaining'] >= app['Storage']):
                allocated = True
                gateway_load[main_gateway] += 1

        # Update resources and calculate metrics if allocated
        if allocated:
            node_resources[target_node]['cpu_remaining'] -= app['CPU']
            node_resources[target_node]['ram_remaining'] -= app['RAM']
            node_resources[target_node]['storage_remaining'] -= app['Storage']
            node_app_counts[target_node] += 1
            results['allocated_count'] += 1
            if graph.nodes[target_node].get('node_type') == 'regular':
                results['simple_node_allocations'] += 1
                results['utilized_simple_nodes'].add(target_node)
            elif graph.nodes[target_node].get('node_type') == 'cloud':
                results['cloud_allocations'] += 1

            results['total_CPU'] += app['CPU']
            results['total_RAM'] += app['RAM']
            results['total_storage'] += app['Storage']

            # Calculate path-based metrics
            try:
                path = nx.shortest_path(graph, source=target_node, target=external_cloud, weight='propagation_delay')
                if len(path) > 1:
                    bandwidth = min(graph.edges[edge]['bandwidth'] for edge in zip(path[:-1], path[1:]))
                    latency = sum(graph.edges[edge]['propagation_delay'] for edge in zip(path[:-1], path[1:]))
                else:
                    bandwidth = config['EXTERNAL_CLOUD']['c_bandwidth']
                    latency = config['EXTERNAL_CLOUD']['c_propagation_delay']
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                bandwidth = config['EXTERNAL_CLOUD']['c_bandwidth']
                latency = config['EXTERNAL_CLOUD']['c_propagation_delay']

            results['total_Latency'] += latency
            results['total_Bandwidth'] += bandwidth

            # Makespan with concurrency
            app_makespan = app['Runtime'] + (app['Msg_Size'] * 8 / bandwidth)
            concurrent_slots = max(1, graph.nodes[target_node]['cpu'] // concurrency_limit)
            if node_app_counts[target_node] <= concurrent_slots:
                node_makespans[target_node] = max(node_makespans[target_node], app_makespan)
            else:
                node_makespans[target_node] += app_makespan

            # Workload, energy, and cost
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

        # Record allocation status
        results['allocation_status'].append({
            'app_id': app_idx,
            'node': target_node if allocated else None,
            'allocated': allocated,
            'requirements': app
        })

    # Finalize results
    results['utilized_simple_nodes_count'] = len(results['utilized_simple_nodes'])
    results['total_Makespan'] = max(node_makespans.values()) if node_makespans else 0.0

    # print(f"Concurrency Limit: {concurrency_limit}")
    # print(f"Node Makespans: {node_makespans}")
    # print(f"Node App Counts: {node_app_counts}")
    # print(f"Gateway Load: {gateway_load}")

    return results

if __name__ == "__main__":
    # This block is optional and can be used for standalone testing
    pass