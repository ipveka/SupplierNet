"""
Supply chain data simulation for SupplierNet.
Generates realistic procurement and manufacturing scenarios with configurable parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class SKUMetadata:
    """Metadata for a single SKU."""
    sku_id: str
    unit_cost: float
    base_demand: float
    procurement_lead_time: float
    manufacturing_lead_time: float
    ordering_cost: float
    seasonality_phase: float  # Phase shift for seasonal pattern

class SupplyChainSimulator:
    """
    Simulates supply chain data for training neural networks.
    
    Generates weekly data including:
    - Demand patterns with seasonality and noise
    - Inventory dynamics with procurement and manufacturing options
    - Cost calculations and optimal supply decisions
    """
    
    def __init__(self, config: Dict):
        """Initialize simulator with configuration parameters."""
        self.config = config
        self.sim_config = config['simulation']
        self.feature_config = config['features']
        
        # Set random seed for reproducibility
        np.random.seed(config['random_seed'])
        
        # Initialize SKU metadata
        self.skus = self._generate_sku_metadata()
        
    def _generate_sku_metadata(self) -> List[SKUMetadata]:
        """Generate metadata for all SKUs."""
        skus = []
        num_skus = self.sim_config['num_skus']
        
        for i in range(num_skus):
            # Generate SKU parameters
            unit_cost = np.random.uniform(*self.sim_config['unit_cost_range'])
            base_demand = np.random.uniform(*self.sim_config['base_demand_range'])
            
            # Lead times with some correlation to demand (higher demand = shorter lead times)
            demand_factor = (base_demand - self.sim_config['base_demand_range'][0]) / \
                          (self.sim_config['base_demand_range'][1] - self.sim_config['base_demand_range'][0])
            
            proc_lt_range = self.sim_config['procurement_lead_time_range']
            proc_lead_time = proc_lt_range[1] - demand_factor * (proc_lt_range[1] - proc_lt_range[0])
            
            manuf_lt_range = self.sim_config['manufacturing_lead_time_range']
            manuf_lead_time = manuf_lt_range[1] - demand_factor * (manuf_lt_range[1] - manuf_lt_range[0])
            
            ordering_cost = np.random.uniform(*self.sim_config['ordering_cost_range'])
            seasonality_phase = np.random.uniform(0, 2 * np.pi)
            
            sku = SKUMetadata(
                sku_id=f"SKU_{i:03d}",
                unit_cost=unit_cost,
                base_demand=base_demand,
                procurement_lead_time=max(1, proc_lead_time),
                manufacturing_lead_time=max(1, manuf_lead_time),
                ordering_cost=ordering_cost,
                seasonality_phase=seasonality_phase
            )
            skus.append(sku)
            
        return skus
    
    def _generate_demand(self, sku: SKUMetadata, weeks: int) -> np.ndarray:
        """Generate demand time series for a SKU."""
        time_points = np.arange(weeks)
        
        # Seasonal component
        seasonal = 1.0
        if self.feature_config['include_seasonality']:
            seasonal = 1 + self.sim_config['seasonality_amplitude'] * \
                      np.sin(2 * np.pi * time_points / 52 + sku.seasonality_phase)
        
        # Trend component
        trend = 1.0
        if self.feature_config['include_trend']:
            trend_rate = np.random.uniform(-0.001, 0.002)  # Small weekly trend
            trend = 1 + trend_rate * time_points
        
        # Base demand with seasonal and trend adjustments
        mean_demand = sku.base_demand * seasonal * trend
        
        # Add noise
        noise_std = self.sim_config['demand_noise_std'] * sku.base_demand
        noise = np.random.normal(0, noise_std, weeks)
        
        # Generate actual demand based on distribution type
        if self.sim_config['demand_distribution'] == 'poisson':
            # For Poisson, use mean_demand as lambda parameter
            demand = np.random.poisson(np.maximum(0.1, mean_demand + noise))
        else:  # normal
            demand = np.maximum(0, mean_demand + noise)
            
        return demand.astype(float)
    
    def _calculate_optimal_order_quantity(self, sku: SKUMetadata, current_inventory: float,
                                        pipeline_inventory: float, forecast_demand: np.ndarray,
                                        supply_method: str) -> float:
        """
        Calculate optimal order quantity using a simple heuristic.
        This serves as the target for neural network training.
        """
        # Lead time based on supply method
        lead_time = sku.procurement_lead_time if supply_method == 'procurement' else sku.manufacturing_lead_time
        
        # Demand during lead time
        lead_time_demand = np.sum(forecast_demand[:int(np.ceil(lead_time))])
        
        # Safety stock
        safety_stock = self.sim_config['safety_stock_weeks'] * np.mean(forecast_demand[:4])
        
        # Target inventory level
        target_inventory = lead_time_demand + safety_stock
        
        # Current position (on-hand + pipeline)
        current_position = current_inventory + pipeline_inventory
        
        # Order quantity
        order_qty = max(0, target_inventory - current_position)
        
        # Cap at maximum order quantity
        max_order = self.sim_config['max_order_quantity_weeks'] * np.mean(forecast_demand[:4])
        order_qty = min(order_qty, max_order)
        
        return order_qty
    
    def _simulate_inventory_dynamics(self, sku: SKUMetadata, demand: np.ndarray) -> Dict:
        """Simulate inventory dynamics for a single SKU."""
        weeks = len(demand)
        
        # Initialize tracking arrays
        inventory = np.zeros(weeks)
        pipeline_proc = np.zeros(weeks)  # Procurement pipeline
        pipeline_manuf = np.zeros(weeks)  # Manufacturing pipeline
        orders_proc = np.zeros(weeks)
        orders_manuf = np.zeros(weeks)
        stockouts = np.zeros(weeks)
        
        # Initial conditions
        initial_inv = self.sim_config['initial_inventory_weeks'] * sku.base_demand
        inventory[0] = initial_inv
        
        # Lead time arrays for tracking deliveries
        proc_deliveries = np.zeros(weeks + int(sku.procurement_lead_time) + 5)
        manuf_deliveries = np.zeros(weeks + int(sku.manufacturing_lead_time) + 5)
        
        for week in range(weeks):
            # Receive deliveries from previous orders
            if week > 0:
                inventory[week] = inventory[week-1] + proc_deliveries[week] + manuf_deliveries[week]
            
            # Meet demand
            demand_met = min(demand[week], inventory[week])
            stockout = demand[week] - demand_met
            inventory[week] -= demand_met
            stockouts[week] = stockout
            
            # Update pipeline inventory
            if week > 0:
                pipeline_proc[week] = max(0, pipeline_proc[week-1] - proc_deliveries[week])
                pipeline_manuf[week] = max(0, pipeline_manuf[week-1] - manuf_deliveries[week])
            
            # Decide on supply method and quantity
            # Simple heuristic: prefer manufacturing if lead time is shorter and cost is lower
            manuf_cost = sku.unit_cost * self.sim_config['production_cost_multiplier']
            proc_cost = sku.unit_cost
            
            # Forecast future demand (simple moving average)
            forecast_window = min(self.feature_config['forecast_horizon'], weeks - week)
            if forecast_window > 0:
                if week + forecast_window < weeks:
                    forecast = demand[week:week + forecast_window]
                else:
                    # Use historical average for forecasting beyond available data
                    hist_avg = np.mean(demand[max(0, week-4):week+1])
                    forecast = np.full(forecast_window, hist_avg)
            else:
                forecast = np.array([sku.base_demand])
            
            # Choose supply method based on cost and lead time
            if (manuf_cost < proc_cost and 
                sku.manufacturing_lead_time <= sku.procurement_lead_time * 1.2):
                supply_method = 'manufacturing'
                order_qty = self._calculate_optimal_order_quantity(
                    sku, inventory[week], pipeline_manuf[week], forecast, supply_method)
                orders_manuf[week] = order_qty
                
                # Schedule delivery
                delivery_week = week + int(np.round(sku.manufacturing_lead_time))
                if delivery_week < len(manuf_deliveries):
                    manuf_deliveries[delivery_week] += order_qty
                    pipeline_manuf[week] += order_qty
            else:
                supply_method = 'procurement'
                order_qty = self._calculate_optimal_order_quantity(
                    sku, inventory[week], pipeline_proc[week], forecast, supply_method)
                orders_proc[week] = order_qty
                
                # Schedule delivery
                delivery_week = week + int(np.round(sku.procurement_lead_time))
                if delivery_week < len(proc_deliveries):
                    proc_deliveries[delivery_week] += order_qty
                    pipeline_proc[week] += order_qty
        
        return {
            'inventory': inventory,
            'pipeline_procurement': pipeline_proc,
            'pipeline_manufacturing': pipeline_manuf,
            'orders_procurement': orders_proc,
            'orders_manufacturing': orders_manuf,
            'stockouts': stockouts,
            'demand_met': demand - stockouts
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset for all SKUs."""
        print(f"Generating dataset for {len(self.skus)} SKUs over {self.sim_config['time_horizon_weeks']} weeks...")
        
        all_data = []
        weeks = self.sim_config['time_horizon_weeks']
        
        for sku_idx, sku in enumerate(self.skus):
            if (sku_idx + 1) % 20 == 0:
                print(f"Processing SKU {sku_idx + 1}/{len(self.skus)}")
            
            # Generate demand
            demand = self._generate_demand(sku, weeks)
            
            # Simulate inventory dynamics
            dynamics = self._simulate_inventory_dynamics(sku, demand)
            
            # Create features for each week
            for week in range(weeks):
                # Historical features (lookback)
                lookback = self.feature_config['lookback_weeks']
                hist_start = max(0, week - lookback)
                
                # Demand history
                hist_demand = demand[hist_start:week] if week > 0 else np.array([sku.base_demand])
                avg_demand = np.mean(hist_demand) if len(hist_demand) > 0 else sku.base_demand
                demand_trend = np.polyfit(range(len(hist_demand)), hist_demand, 1)[0] if len(hist_demand) > 1 else 0
                demand_volatility = np.std(hist_demand) if len(hist_demand) > 1 else 0
                
                # Current state
                current_inventory = dynamics['inventory'][week]
                pipeline_proc = dynamics['pipeline_procurement'][week]
                pipeline_manuf = dynamics['pipeline_manufacturing'][week]
                
                # Time features
                week_of_year = week % 52
                seasonality_sin = np.sin(2 * np.pi * week_of_year / 52)
                seasonality_cos = np.cos(2 * np.pi * week_of_year / 52)
                
                # Target: total order quantity (procurement + manufacturing)
                target_qty = dynamics['orders_procurement'][week] + dynamics['orders_manufacturing'][week]
                
                # Supply method indicator (1 for manufacturing, 0 for procurement)
                supply_method = 1 if dynamics['orders_manufacturing'][week] > 0 else 0
                
                row = {
                    'sku_id': sku.sku_id,
                    'week': week,
                    'current_inventory': current_inventory,
                    'pipeline_procurement': pipeline_proc,
                    'pipeline_manufacturing': pipeline_manuf,
                    'avg_demand_lookback': avg_demand,
                    'demand_trend': demand_trend,
                    'demand_volatility': demand_volatility,
                    'procurement_lead_time': sku.procurement_lead_time,
                    'manufacturing_lead_time': sku.manufacturing_lead_time,
                    'unit_cost': sku.unit_cost,
                    'seasonality_sin': seasonality_sin,
                    'seasonality_cos': seasonality_cos,
                    'target_quantity': target_qty,
                    'supply_method': supply_method,
                    'actual_demand': demand[week],
                    'stockout': dynamics['stockouts'][week],
                }
                
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        print(f"Generated dataset with {len(df)} records")
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """Save dataset to CSV file."""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def get_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics of the dataset."""
        summary = {
            'total_records': len(df),
            'num_skus': df['sku_id'].nunique(),
            'time_horizon': df['week'].max() + 1,
            'avg_demand': df['actual_demand'].mean(),
            'avg_target_quantity': df['target_quantity'].mean(),
            'stockout_rate': (df['stockout'] > 0).mean(),
            'manufacturing_preference': df['supply_method'].mean(),
            'feature_correlations': df.select_dtypes(include=[np.number]).corr()['target_quantity'].sort_values(ascending=False)
        }
        return summary
