import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import configparser
from pathlib import Path
from neighbor_allocation import run_neighbor_allocation
from pso_allocation import run_pso_allocation  # Import PSO Allocation
from save_csv import save_to_csv


def read_config(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Reads configuration from a file and returns it as a dictionary with robust parsing."""
    config = configparser.ConfigParser()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file '{file_path}' not found.")
    config.read(file_path)
    if not config.sections():
        raise ValueError(f"Config file '{file_path}' is empty or malformed.")
    
    def parse_value(value: str) -> Any:
        value = value.split(';')[0].strip()
        try:
            return int(value) if value.isdigit() else float(value)
        except ValueError:
            return value
    
    return {section: {key: parse_value(value) for key, value in config.items(section)} for section in config.sections()}


def generate_graph(node_count: int, edge_prob: float, config: Dict[str, Any]) -> str:
    """Generates a connected Erdős-Rényi graph with the specified number of nodes and one external cloud node."""
    fog_node_count = node_count
    edge_prob = edge_prob or config['FOG_ATTRIBUTES']['f_edge_prob']
    G = nx.erdos_renyi_graph(fog_node_count, edge_prob)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(fog_node_count, edge_prob)

    # Initialize all nodes with default attributes and node_type='regular'
    for node in G.nodes:
        G.nodes[node]['cpu'] = random.randint(config['FOG_ATTRIBUTES']['f_cpu_min'], config['FOG_ATTRIBUTES']['f_cpu_max'])
        G.nodes[node]['ram'] = random.randint(config['FOG_ATTRIBUTES']['f_ram_min'], config['FOG_ATTRIBUTES']['f_ram_max'])
        G.nodes[node]['storage'] = random.randint(config['FOG_ATTRIBUTES']['f_storage_min'], config['FOG_ATTRIBUTES']['f_storage_max'])
        G.nodes[node]['bandwidth'] = max(
            random.uniform(config['FOG_ATTRIBUTES']['f_bandwidth_min'], config['FOG_ATTRIBUTES']['f_bandwidth_max']),
            0.1
        )
        G.nodes[node]['node_type'] = 'regular'  # Default to regular
        print(f"Node {node} initialized with resources: CPU={G.nodes[node]['cpu']}, RAM={G.nodes[node]['ram']}, Storage={G.nodes[node]['storage']}, Bandwidth={G.nodes[node]['bandwidth']}")
    
    # Set edge attributes
    for edge in G.edges:
        G.edges[edge]['propagation_delay'] = round(random.uniform(config['FOG_ATTRIBUTES']['f_propagation_delay_min'], config['FOG_ATTRIBUTES']['f_propagation_delay_max']), 3)
        G.edges[edge]['bandwidth'] = round(random.uniform(config['FOG_ATTRIBUTES']['f_bandwidth_min'], config['FOG_ATTRIBUTES']['f_bandwidth_max']), 3)

    pos = nx.spring_layout(G)
    Path('Topologies').mkdir(exist_ok=True)
    graph_base_filename = f'Topologies/topology_n{fog_node_count}'
    graph_file_id = 1
    while os.path.exists(f'{graph_base_filename}_{graph_file_id}.graphml'):
        graph_file_id += 1
    graphml_file = f'{graph_base_filename}_{graph_file_id}.graphml'

    # Identify and tag main gateway
    centrality = nx.betweenness_centrality(G)
    main_gateway = max(centrality.items(), key=lambda x: x[1])[0]
    G.nodes[main_gateway]['node_type'] = 'gateway'  # Tag main gateway

    # Identify and tag additional gateway nodes
    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(centrality, key=centrality.get)
    gateway_percentage = config['FOG_ATTRIBUTES']['f_gateway_percentage']
    gateway_node_count = max(1, int(gateway_percentage * fog_node_count))
    gateway_list = sorted_nodes[:gateway_node_count]
    for node in gateway_list:
        if node != main_gateway:  # Avoid overwriting main gateway
            G.nodes[node]['node_type'] = 'gateway'  # Tag as gateway

    # Add the cloud node
    cloud = max(G.nodes) + 1
    G.add_node(cloud, 
               cpu=config['EXTERNAL_CLOUD']['c_cpu'],
               ram=config['EXTERNAL_CLOUD']['c_ram'],
               storage=config['EXTERNAL_CLOUD']['c_storage'],
               bandwidth=config['EXTERNAL_CLOUD']['c_bandwidth'],
               runtime=config['EXTERNAL_CLOUD']['c_runtime'],
               node_type='cloud')
    print(f"Cloud node added: {cloud}, Attributes: {G.nodes[cloud]}")
    G.add_edge(main_gateway, cloud, 
               propagation_delay=config['EXTERNAL_CLOUD']['c_propagation_delay'], 
               bandwidth=config['EXTERNAL_CLOUD']['c_bandwidth'])
    pos[cloud] = (0.5, 0.5)

    # Save the graph after adding all nodes and attributes
    nx.write_graphml(G, graphml_file)
    print(f"Graph saved to {graphml_file}.\n") 

    # Ensure all non-gateway nodes in a graph G are connected to at least one gateway node
    for node in G.nodes:
        if node not in gateway_list and node != main_gateway and node != cloud:
            if not any(neighbor in gateway_list for neighbor in G.neighbors(node)):
                gateway_node = random.choice(gateway_list)
                G.add_edge(node, gateway_node, 
                           propagation_delay=random.uniform(config['FOG_ATTRIBUTES']['f_propagation_delay_min'], config['FOG_ATTRIBUTES']['f_propagation_delay_max']),
                           bandwidth=random.uniform(config['FOG_ATTRIBUTES']['f_bandwidth_min'], config['FOG_ATTRIBUTES']['f_bandwidth_max']))

    # Visualization (unchanged)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n not in gateway_list and n != main_gateway and n != cloud],
                           node_color='skyblue', node_shape='o', node_size=500, label='Regular Fog Nodes')
    nx.draw_networkx_nodes(G, pos, nodelist=gateway_list, node_color='lightgreen', node_shape='s', node_size=600, label='Gateway Nodes')
    nx.draw_networkx_nodes(G, pos, nodelist=[main_gateway], node_color='limegreen', node_shape='s', node_size=900, label='Main Gateway')
    nx.draw_networkx_nodes(G, pos, nodelist=[cloud], node_color='salmon', node_shape='o', node_size=1000, label='Cloud Node')
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if (u, v) != (main_gateway, cloud)], edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=[(main_gateway, cloud)], width=2.0)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.title('Erdős-Rényi - Fog Computing Topology')
    plt.savefig(f'{graph_base_filename}_{graph_file_id}.png')
    plt.close()

    return graphml_file

def load_graph(graphml_file: str) -> nx.Graph:
    if not os.path.exists(graphml_file):
        raise FileNotFoundError(f"Graph file '{graphml_file}' not found.")
    G = nx.read_graphml(graphml_file)

    # Convert node attributes to correct types
    for node in G.nodes:
        G.nodes[node]['cpu'] = int(float(G.nodes[node].get('cpu', 0)))
        G.nodes[node]['ram'] = int(float(G.nodes[node].get('ram', 0)))
        G.nodes[node]['storage'] = int(float(G.nodes[node].get('storage', 0)))
        G.nodes[node]['cpu_remaining'] = G.nodes[node]['cpu']
        G.nodes[node]['ram_remaining'] = G.nodes[node]['ram']
        G.nodes[node]['storage_remaining'] = G.nodes[node]['storage']
        G.nodes[node]['node_type'] = G.nodes[node].get('node_type', 'regular')  # Preserve node_type
        if G.nodes[node]['node_type'] == 'cloud':  # Ensure cloud node is recognized
            pass
        G.nodes[node]['bandwidth'] = int(float(G.nodes[node].get('bandwidth', 0)))  # Preserve if present
        G.nodes[node]['runtime'] = int(float(G.nodes[node].get('runtime', 0)))  # Preserve if present

    # Convert edge attributes to correct types
    for edge in G.edges:
        G.edges[edge]['propagation_delay'] = float(G.edges[edge].get('propagation_delay', 0.0))
        G.edges[edge]['bandwidth'] = float(G.edges[edge].get('bandwidth', 0.0))

    # Ensure the graph object is returned
    return G

def validate_graph(graph: nx.Graph):
    """Ensure the graph has valid nodes and edges."""
    if not graph.nodes:
        raise ValueError("The graph has no nodes.")
    if not graph.edges:
        raise ValueError("The graph has no edges.")
    print(f"Graph validation passed: {len(graph.nodes)} nodes, {len(graph.edges)} edges.")

if __name__ == '__main__':
    print('\nWelcome to FogSim-NX - A Simulation Framework for Fog Computing.\n')
    print('This program simulates fog computing environments, using Neighbor-Aware and PSO algorithms to model resource allocation on a NetworkX-based Erdős-Rényi topology, optimized for makespan, energy, and cost.\n')
    try:
        config = read_config('config.ini')  # Ensure config is always loaded
        while True:
            use_existing_topology = input('Do you want to use an existing topology (y), create new (n), or exit (e)? [y/n/e]: ').strip().lower()
            if use_existing_topology == 'y':
                topology_folder = Path('Topologies')
                if not topology_folder.exists():
                    print("Error: No 'Topologies' folder found. Please generate a topology first.")
                    continue  # Loop back to the input prompt
                graphml_files = list(topology_folder.glob('*.graphml'))
                if not graphml_files:
                    print("Error: No .graphml files found in the 'Topologies' folder. Please generate a topology first.")
                    continue  # Loop back to the input prompt
                print("Available topologies:")
                for i, file in enumerate(graphml_files, start=1):
                    print(f"{i}: {file.name}")
                selected_index = int(input('Select a topology by number: ')) - 1
                if selected_index < 0 or selected_index >= len(graphml_files):
                    print("Error: Invalid selection.")
                    continue  # Loop back to the input prompt
                graphml_file = graphml_files[selected_index]
                break
            elif use_existing_topology == 'n':
                num_nodes = int(input('Creating new Topology - Enter the number of nodes : '))
                print(f"Generating new topology with {num_nodes} nodes...")
                graphml_file = generate_graph(num_nodes, config['FOG_ATTRIBUTES']['f_edge_prob'], config)
                break
            elif use_existing_topology == 'e':
                print("Exiting the program.")
                exit(0)
            else:
                print("Invalid input. Please enter 'y', 'n', or 'e'.")
        
        # Prompt the user to input the number of applications to allocate and the allocation method
        app_count = int(input('Enter the number of applications to be allocated: '))
        allocation_choice = int(input('Choose allocation method (1-Neighbor-Aware Method, 2-PSO Algorithm): '))
    except ValueError:
        print("Error: Please enter valid integer values.")
        exit(1)

    # Load the selected topology file into a NetworkX graph object
    print(f"Loading topology from {graphml_file}...")
    graph = load_graph(graphml_file)

    validate_graph(graph)  # Validate the graph before proceeding

    # Identify the external cloud node in the graph
    cloud_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get('node_type') == 'cloud']
    if not cloud_nodes:
        raise ValueError("External cloud node not found in the graph. Check topology generation and loading.")
    external_cloud = cloud_nodes[0]  # Assuming a single cloud node

    # Execute the selected allocation method
    if allocation_choice == 1:
        # Run the Neighbor-Aware Allocation algorithm
        print("Running Neighbor-Aware Allocation...")
        results = run_neighbor_allocation(graph, app_count, config, external_cloud)  # Pass external_cloud
    elif allocation_choice == 2:
        # Run the PSO Algorithm for allocation
        print("Running PSO Algorithm Allocation...")
        results = run_pso_allocation(graph, app_count, config, external_cloud)  # Pass external_cloud
    else:
        print("Invalid choice. Exiting.")
        exit(1)

    if results is None:
        print("Error: Allocation function returned None. Exiting.")
        exit(1)

    # Print allocation status for each application
    for app_details in results['allocation_status']:
        app_id = app_details['app_id']
        if app_details['allocated']:
            print(f"Application {app_id} {app_details['requirements']} successfully allocated to Node {app_details['node']}.")
        else:
            print(f"Application {app_id} {app_details['requirements']} could not be allocated.")

    # Print results
    print(f"\nSimulation Results:")
    print(f"Allocated {results['allocated_count']} out of {app_count} applications.")    
    print(f"Applications allocated to fog nodes: {results['fog_node_allocations']}")
    print(f"Applications allocated to the Cloud: {results['cloud_allocations']}")
    print(f"\nNumber of fog nodes utilized: {results['utilized_fog_nodes_count']}")
    print(f"Total CPU Used: {results['total_CPU']} MIPS")
    print(f"Total RAM Used: {int(results['total_RAM'] / 1024)} GB")
    print(f"Total Storage Used: {int(results['total_storage'] / 1024)} GB")
    print(f"Total Bandwidth Used: {int(results['total_Bandwidth'] / 1000)} Gbps")
    print(f"Total Latency: {int(results['total_Latency'])} seconds")
    print(f"Total Makespan: {int(results['total_Makespan'])} seconds")
    print(f"Total Workload: {int(results['total_Workload'])} WU")
    print(f"Total Energy Consumption: {int(results['total_Energy'])} Watts")
    print(f"Total Cost: €{int(results['total_Cost'])}\n")

    # Save results to CSV in Results folder
    allocation_method = "Neighbor-Aware" if allocation_choice == 1 else "PSO"
    save_to_csv(
        results={
            **results,
            'total_RAM': int(results['total_RAM']),
            'total_storage': int(results['total_storage']),
            'total_Bandwidth': int(results['total_Bandwidth']),
            'total_Latency': int(results['total_Latency']),
            'total_Makespan': int(results['total_Makespan']),
            'total_Workload': int(results['total_Workload']),
            'total_Energy': int(results['total_Energy']),
            'total_Cost': int(results['total_Cost'])
        },
        topology_nodes_count=len(graph.nodes),
        topology_file=graphml_file,  # Pass graphml_file directly as a string
        app_count=app_count,
        allocation_method=allocation_method
    )