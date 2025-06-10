# FogSim-NX: A Fog Computing Simulation Framework

A comprehensive NetworkX-based simulation framework for fog computing environments that models resource allocation using advanced algorithms to optimize makespan, energy consumption, and cost efficiency.

## Features

- Topology Generation: Creates realistic Erdős-Rényi fog computing topologies with configurable parameters
- Dual Allocation Algorithms: 
  - Neighbor-Aware Method for intelligent resource allocation
  - Particle Swarm Optimization (PSO) Algorithm for optimized allocation
- Multi-objective Optimization: Simultaneously optimizes makespan, energy consumption, and cost
- Comprehensive Metrics: Tracks CPU, RAM, storage, bandwidth, latency, workload, energy, and cost
- Visualization: Automatic topology visualization with node type differentiation
- Results Export: CSV export functionality for analysis and comparison
- Configurable Parameters: Extensive configuration through INI files

## Architecture

### Network Topology
- Regular Nodes: Standard fog computing nodes with limited resources
- Gateway Nodes: Intermediate nodes that forward applications to optimal locations
- Main Gateway: Central node with highest betweenness centrality
- External Cloud: Unlimited resources for overflow capacity

### Allocation Algorithms

#### 1. Neighbor-Aware Method
- Gateways intelligently forward applications to neighboring nodes
- Minimizes makespan increase through strategic placement
- Falls back to cloud resources when local allocation fails
- Load balancing across gateway nodes

#### 2. Particle Swarm Optimization (PSO)
- Global optimization using swarm intelligence
- Multi-objective fitness function considering:
  - Resource utilization efficiency
  - Latency minimization
  - Energy consumption
  - Cost optimization
  - Cloud usage penalties
- Iterative improvement with configurable parameters

## Requirements

```bash
pip install networkx matplotlib numpy configparser pathlib
```

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/yourusername/fogsim-nx.git
cd fogsim-nx
```

2. Configure simulation parameters (optional)
Edit `config.ini` to customize topology and application requirements.

3. Run the simulation
```bash
python FogSim-NX.py
```

4. Follow the interactive prompts
- Choose to create new topology or use existing
- Specify number of applications
- Select allocation method (Neighbor-Aware or PSO)

## Configuration

The `config.ini` file contains comprehensive configuration options:

### Topology Attributes
- Edge Probability: Connectivity density between nodes
- Gateway Percentage: Proportion of nodes designated as gateways
- Resource Ranges: CPU (MIPS), RAM (MB), Storage (MB), Bandwidth (Mbps)
- Network Delays: Propagation delay ranges

### Application Requirements
- Resource Demands: CPU, RAM, Storage requirements
- Runtime Parameters: Execution time ranges
- Message Size: Communication overhead

### Energy & Cost Models
- Power Consumption: Idle, transmission, and processing power
- Unit Costs: Energy, CPU, RAM, and storage pricing

## Output Metrics

The simulation provides comprehensive performance metrics:

### Resource Utilization
- Total CPU, RAM, Storage, and Bandwidth usage
- Node utilization distribution
- Cloud vs. fog allocation ratios

### Performance Metrics
- Makespan: Total execution time considering concurrency
- Latency: Network communication delays
- Workload: Composite resource demand metric

### Efficiency Metrics
- Energy Consumption: Total power usage across all nodes
- Cost: Economic efficiency including resource and energy costs
- Allocation Success Rate: Percentage of successfully allocated applications

## Project Structure

```
fogsim-nx/
├── FogSim-NX.py              # Main simulation framework
├── neighbor_allocation.py     # Neighbor-aware allocation algorithm
├── pso_allocation.py         # PSO optimization algorithm
├── save_csv.py               # Results export functionality
├── config.ini                # Configuration parameters
├── Topology/                 # Generated topology files
│   ├── topology_n*.graphml   # NetworkX graph files
│   └── topology_n*.png       # Visualization images
└── Results/                  # Simulation results
    └── simulation_results_*.csv
```

## Research Applications

This framework is designed for research in:

- Fog Computing Resource Management
- Edge-Cloud Continuum Optimization
- Multi-objective Scheduling Algorithms
- Energy-Efficient Computing
- Network Topology Impact Analysis

## Performance Comparison

The framework enables comparative analysis between allocation methods:

| Metric | Neighbor-Aware | PSO Algorithm |
|--------|---------------|---------------|
| Speed | Fast execution | Slower but optimized |
| Optimality | Good local optimization | Global optimization |
| Scalability | Linear complexity | Configurable complexity |
| Use Case | Real-time allocation | Batch optimization |

## Advanced Usage

### Custom Topologies
```python
# Load existing topology
graph = load_graph('Topology/topology_n50_1.graphml')

# Run custom allocation
results = run_neighbor_allocation(graph, app_count=100, config=config, external_cloud=cloud_node)
```

### Batch Experiments
```python
# Automated parameter sweeps
for nodes in [10, 20, 50, 100]:
    for apps in [50, 100, 200]:
        # Run simulation with different parameters
        topology = generate_graph(nodes, edge_prob=0.3, config=config)
        # Process results...
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use FogSim-NX in your research, please cite:

```bibtex
@software{fogsim_nx,
  title={FogSim-NX: A NetworkX-based Fog Computing Simulation Framework},
  author={Stefanos Nikou},
  year={2025},
  url={https://github.com/stefanos8569/FogSim-NX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Known Issues

- Large topologies (>1000 nodes) may require increased memory allocation
- PSO convergence time scales with particle count and iteration limits
- Visualization performance degrades with very dense topologies

## Future Enhancements

- [ ] Support for heterogeneous device types
- [ ] Dynamic topology changes during simulation
- [ ] Machine learning-based allocation strategies
- [ ] Real-time monitoring and adaptation
- [ ] Integration with containerized applications
- [ ] Support for federated learning workloads

## Support

For questions, issues, or contributions:

- Email: stefanos2077@gmail.com

---

FogSim-NX - Enabling advanced research in fog computing through comprehensive simulation and optimization.