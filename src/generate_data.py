"""
Data generation pipeline for SupplierNet.
Handles supply chain data simulation with configurable parameters.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

# Local imports
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import get_config, set_random_seeds
from data.simulation import SupplyChainSimulator

def generate_supply_chain_data(config: Dict, output_path: str, force_regenerate: bool = False) -> str:
    """
    Generate supply chain dataset using the simulator.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the generated dataset
        force_regenerate: Force regeneration even if file exists
        
    Returns:
        Path to the generated dataset file
    """
    print("=" * 60)
    print("SupplierNet Data Generation Pipeline")
    print("=" * 60)
    
    # Check if dataset already exists
    if os.path.exists(output_path) and not force_regenerate:
        print(f"Dataset already exists at {output_path}")
        print("Use --force-regenerate to create a new dataset")
        return output_path
    
    # Set random seeds for reproducibility
    set_random_seeds(config['random_seed'])
    
    # Initialize simulator
    print(f"Initializing simulator with {config['simulation']['num_skus']} SKUs...")
    simulator = SupplyChainSimulator(config)
    
    # Generate dataset
    print("Generating supply chain dataset...")
    df = simulator.generate_dataset()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save dataset
    simulator.save_dataset(df, output_path)
    
    # Generate and display summary
    summary = simulator.get_dataset_summary(df)
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Dataset Statistics:")
    print(f"  - Total records: {summary['total_records']:,}")
    print(f"  - Number of SKUs: {summary['num_skus']}")
    print(f"  - Time horizon: {summary['time_horizon']} weeks")
    print(f"  - Average demand: {summary['avg_demand']:.2f}")
    print(f"  - Average target quantity: {summary['avg_target_quantity']:.2f}")
    print(f"  - Stockout rate: {summary['stockout_rate']:.3f}")
    print(f"  - Manufacturing preference: {summary['manufacturing_preference']:.3f}")
    
    print(f"\nConfiguration Used:")
    print(f"  - Demand distribution: {config['simulation']['demand_distribution']}")
    print(f"  - Seasonality amplitude: {config['simulation']['seasonality_amplitude']}")
    print(f"  - Procurement lead time: {config['simulation']['procurement_lead_time_range']}")
    print(f"  - Manufacturing lead time: {config['simulation']['manufacturing_lead_time_range']}")
    print(f"  - Holding cost rate: {config['simulation']['holding_cost_rate']}")
    print(f"  - Shortage penalty: {config['simulation']['shortage_penalty_multiplier']}x")
    
    print(f"\nOutput:")
    print(f"  - Dataset saved to: {output_path}")
    print(f"  - File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Save generation metadata
    metadata_path = output_path.replace('.csv', '_metadata.json')
    metadata = {
        'generation_config': config,
        'dataset_summary': summary,
        'output_path': output_path,
        'file_size_mb': os.path.getsize(output_path) / 1024 / 1024
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"  - Metadata saved to: {metadata_path}")
    print("=" * 60)
    
    return output_path

def generate_multiple_datasets(base_config: Dict, variations: Dict, output_dir: str = "data/datasets") -> Dict[str, str]:
    """
    Generate multiple datasets with different configurations.
    
    Args:
        base_config: Base configuration
        variations: Dictionary of configuration variations
        output_dir: Directory to save datasets
        
    Returns:
        Dictionary mapping variation names to dataset paths
    """
    print("Generating multiple dataset variations...")
    
    generated_datasets = {}
    
    for variation_name, variation_config in variations.items():
        print(f"\nGenerating variation: {variation_name}")
        
        # Merge base config with variation
        config = base_config.copy()
        for key, value in variation_config.items():
            if '.' in key:
                # Handle nested keys like 'simulation.num_skus'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        # Generate dataset
        output_path = os.path.join(output_dir, f"{variation_name}_dataset.csv")
        dataset_path = generate_supply_chain_data(config, output_path, force_regenerate=True)
        generated_datasets[variation_name] = dataset_path
    
    return generated_datasets

def main():
    """Main data generation script with command line arguments."""
    parser = argparse.ArgumentParser(description='Generate SupplierNet supply chain datasets')
    parser.add_argument('--output-path', type=str, default='data/supply_chain_dataset.csv',
                       help='Path to save the generated dataset')
    parser.add_argument('--config-overrides', type=str, default='{}',
                       help='JSON string with config overrides')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration even if dataset exists')
    parser.add_argument('--quick-test', action='store_true',
                       help='Generate smaller dataset for quick testing')
    parser.add_argument('--multiple-variations', action='store_true',
                       help='Generate multiple dataset variations')
    parser.add_argument('--output-dir', type=str, default='data/datasets',
                       help='Directory for multiple dataset variations')
    
    args = parser.parse_args()
    
    # Load base configuration
    config = get_config()
    
    # Apply quick test overrides
    if args.quick_test:
        print("Running in quick test mode...")
        config['simulation']['num_skus'] = 20
        config['simulation']['time_horizon_weeks'] = 52
    
    # Apply config overrides
    try:
        overrides = json.loads(args.config_overrides)
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'simulation.num_skus'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        if overrides:
            print(f"Applied config overrides: {overrides}")
    except json.JSONDecodeError as e:
        print(f"Error parsing config overrides: {e}")
        return
    
    try:
        if args.multiple_variations:
            # Define dataset variations
            variations = {
                'small': {'simulation.num_skus': 50, 'simulation.time_horizon_weeks': 52},
                'medium': {'simulation.num_skus': 100, 'simulation.time_horizon_weeks': 104},
                'large': {'simulation.num_skus': 200, 'simulation.time_horizon_weeks': 156},
                'high_seasonality': {'simulation.seasonality_amplitude': 0.5},
                'low_seasonality': {'simulation.seasonality_amplitude': 0.1},
                'long_lead_times': {
                    'simulation.procurement_lead_time_range': [4, 12],
                    'simulation.manufacturing_lead_time_range': [2, 8]
                },
                'short_lead_times': {
                    'simulation.procurement_lead_time_range': [1, 4],
                    'simulation.manufacturing_lead_time_range': [1, 2]
                }
            }
            
            generated_datasets = generate_multiple_datasets(config, variations, args.output_dir)
            
            print(f"\nGenerated {len(generated_datasets)} dataset variations:")
            for name, path in generated_datasets.items():
                print(f"  - {name}: {path}")
        
        else:
            # Generate single dataset
            dataset_path = generate_supply_chain_data(
                config=config,
                output_path=args.output_path,
                force_regenerate=args.force_regenerate
            )
            
            print(f"\nDataset generation completed successfully!")
            print(f"Dataset available at: {dataset_path}")
        
    except Exception as e:
        print(f"ERROR: Data generation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
