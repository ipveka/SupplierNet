"""
Visualization utilities for SupplierNet model evaluation and analysis.
Provides comprehensive plots for predictions, costs, service levels, and business insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

class SupplyChainVisualizer:
    """
    Comprehensive visualization suite for supply chain optimization results.
    
    Features:
    - Prediction accuracy plots
    - Cost breakdown analysis
    - Service level visualizations
    - Time series analysis
    - SKU-level performance comparisons
    """
    
    def __init__(self, config: Dict, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
            figsize: Default figure size
        """
        self.config = config
        self.eval_config = config['evaluation']
        self.figsize = figsize
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F18F01',
            'info': '#2E86AB'
        }
    
    def plot_prediction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                title: str = "Prediction Accuracy", 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create prediction accuracy visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        
        # Scatter plot: Predicted vs True
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color=self.colors['primary'])
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs True Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to the plot
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color=self.colors['secondary'])
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color=self.colors['accent'], edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for residuals normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction accuracy plot saved to {save_path}")
        
        return fig
    
    def plot_cost_breakdown(self, cost_metrics: Dict, title: str = "Cost Breakdown Analysis",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize cost breakdown and analysis.
        
        Args:
            cost_metrics: Cost metrics dictionary
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        
        # Cost breakdown pie chart - improved
        cost_breakdown = cost_metrics['cost_breakdown']
        labels = [label.title() for label in cost_breakdown.keys()]
        sizes = list(cost_breakdown.values())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Create exploded pie chart with better styling
        explode = (0.05, 0.05, 0.05, 0.05)  # explode all slices slightly
        wedges, texts, autotexts = axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', 
                                                 colors=colors, startangle=90, explode=explode,
                                                 shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[0, 0].set_title('Cost Breakdown by Component', fontsize=14, fontweight='bold')
        
        # Cost components bar chart - improved
        cost_components = {
            'Holding': cost_metrics['holding_cost_total'],
            'Shortage': cost_metrics['shortage_cost_total'],
            'Ordering': cost_metrics['ordering_cost_total'],
            'Production': cost_metrics['production_cost_total']
        }
        
        bars = axes[0, 1].bar(cost_components.keys(), cost_components.values(), 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        axes[0, 1].set_title('Total Cost by Component', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=11)
        axes[0, 1].tick_params(axis='y', labelsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars with better formatting
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'${height:,.0f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
        
        # Cost efficiency metrics - improved display
        total_cost = cost_metrics['total_cost']
        avg_cost = cost_metrics['avg_cost_per_period']
        
        # Create a more professional metrics display
        axes[1, 0].axis('off')
        
        # Main metrics
        axes[1, 0].text(0.5, 0.85, 'Cost Summary', ha='center', va='center', 
                       fontsize=16, fontweight='bold', transform=axes[1, 0].transAxes)
        
        axes[1, 0].text(0.5, 0.7, f'Total Cost: ${total_cost:,.0f}', ha='center', va='center',
                       fontsize=14, fontweight='bold', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        axes[1, 0].text(0.5, 0.55, f'Avg Cost/Period: ${avg_cost:,.0f}', ha='center', va='center',
                       fontsize=12, transform=axes[1, 0].transAxes)
        
        # Cost breakdown percentages
        y_pos = 0.4
        for component, percentage in cost_breakdown.items():
            axes[1, 0].text(0.5, y_pos, f'{component.title()}: {percentage:.1%}', 
                           ha='center', va='center', fontsize=11, 
                           transform=axes[1, 0].transAxes)
            y_pos -= 0.08
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Cost Summary')
        
        # Cost trend (if multiple periods available)
        axes[1, 1].bar(['Total Cost'], [total_cost], color=self.colors['primary'])
        axes[1, 1].set_title('Total Cost Overview')
        axes[1, 1].set_ylabel('Cost ($)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost breakdown plot saved to {save_path}")
        
        return fig
    
    def plot_service_level_analysis(self, service_metrics: Dict, 
                                   title: str = "Service Level Analysis",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize service level performance.
        
        Args:
            service_metrics: Service level metrics dictionary
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        
        # Service level gauge - improved
        avg_service = service_metrics['avg_service_level']
        target_service = self.eval_config['service_level_threshold']
        
        # Create professional gauge chart
        ax_gauge = axes[0, 0]
        
        # Create semicircle gauge
        angles = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax_gauge.plot(angles, np.ones_like(angles), 'lightgray', linewidth=8, alpha=0.3)
        
        # Service level arc
        service_angle = avg_service * np.pi
        service_angles = np.linspace(0, service_angle, int(avg_service * 100))
        
        # Color based on performance
        if avg_service >= target_service:
            gauge_color = '#2E8B57'  # Sea green
        elif avg_service >= target_service * 0.8:
            gauge_color = '#FFD700'  # Gold
        else:
            gauge_color = '#DC143C'  # Crimson
            
        ax_gauge.plot(service_angles, np.ones_like(service_angles), 
                     color=gauge_color, linewidth=8, alpha=0.8)
        
        # Add target marker
        target_angle = target_service * np.pi
        ax_gauge.plot([target_angle, target_angle], [0.8, 1.2], 'red', 
                     linewidth=3, label=f'Target: {target_service:.0%}')
        
        # Center text
        ax_gauge.text(np.pi/2, 0.5, f'{avg_service:.1%}', ha='center', va='center', 
                     fontsize=24, fontweight='bold', color=gauge_color)
        ax_gauge.text(np.pi/2, 0.3, 'Service Level', ha='center', va='center', 
                     fontsize=12, color='gray')
        
        # Styling
        ax_gauge.set_ylim(0, 1.3)
        ax_gauge.set_xlim(-0.1, np.pi + 0.1)
        ax_gauge.set_title('Service Level Performance', fontsize=14, fontweight='bold')
        ax_gauge.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax_gauge.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
        ax_gauge.set_yticks([])
        ax_gauge.legend(loc='upper right', fontsize=10)
        
        # Service level distribution - improved
        service_data = [
            service_metrics['min_service_level'],
            service_metrics['service_level_5th_percentile'],
            service_metrics['median_service_level'],
            service_metrics['avg_service_level'],
            service_metrics['service_level_95th_percentile']
        ]
        
        labels = ['Min', '5th %ile', 'Median', 'Mean', '95th %ile']
        colors_gradient = ['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#32CD32']
        
        bars = axes[0, 1].bar(labels, service_data, color=colors_gradient, 
                             alpha=0.8, edgecolor='black', linewidth=1.2)
        axes[0, 1].axhline(y=target_service, color='red', linestyle='--', 
                          linewidth=2, label=f'Target: {target_service:.0%}')
        axes[0, 1].set_title('Service Level Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Service Level', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].tick_params(axis='both', labelsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0, 1].legend(fontsize=10)
        
        # Add value labels with better formatting
        for bar, value in zip(bars, service_data):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                           f'{value:.1%}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
        
        # Stockout analysis
        stockout_freq = service_metrics['stockout_frequency']
        perfect_service = service_metrics['perfect_service_rate']
        
        stockout_data = {
            'Stockout Events': stockout_freq,
            'Perfect Service': perfect_service,
            'Partial Service': 1 - stockout_freq - perfect_service
        }
        
        # Professional colors for service analysis
        service_colors = ['#DC143C', '#2E8B57', '#FFD700']  # Red, Green, Gold
        explode = (0.1, 0.05, 0.02)  # Explode stockouts more
        
        wedges, texts, autotexts = axes[1, 0].pie(stockout_data.values(), 
                                                 labels=stockout_data.keys(), 
                                                 autopct='%1.1f%%', startangle=90,
                                                 colors=service_colors, explode=explode,
                                                 shadow=True, 
                                                 textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1, 0].set_title('Service Performance Breakdown', fontsize=14, fontweight='bold')
        
        # Service metrics summary
        metrics_text = f"""
        Service Level Metrics:
        
        • Average: {avg_service:.1%}
        • Target: {target_service:.1%}
        • Achievement: {avg_service/target_service:.1%}
        
        • Stockout Frequency: {stockout_freq:.1%}
        • Perfect Service Rate: {perfect_service:.1%}
        • Service Level Std: {service_metrics['service_level_std']:.3f}
        
        Performance: {'✓ GOOD' if avg_service >= target_service else '⚠ NEEDS IMPROVEMENT'}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Service Level Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Service level analysis plot saved to {save_path}")
        
        return fig
    
    def plot_sku_performance_comparison(self, sku_analysis: Dict, top_n: int = 10,
                                      title: str = "SKU Performance Comparison",
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare performance across SKUs.
        
        Args:
            sku_analysis: SKU analysis dictionary
            top_n: Number of top/bottom SKUs to show
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        sku_metrics = sku_analysis['sku_metrics']
        
        # Extract metrics for all SKUs
        sku_data = []
        for sku_id, metrics in sku_metrics.items():
            sku_data.append({
                'sku_id': sku_id,
                'mae': metrics['prediction_metrics']['mae'],
                'r2': metrics['prediction_metrics']['r2'],
                'service_level': metrics['service_metrics']['avg_service_level'],
                'total_cost': metrics['cost_metrics']['total_cost'],
                'avg_demand': metrics['avg_demand']
            })
        
        sku_df = pd.DataFrame(sku_data)
        
        # Top/Bottom SKUs by MAE
        sku_df_sorted = sku_df.sort_values('mae')
        top_skus = sku_df_sorted.head(top_n)
        bottom_skus = sku_df_sorted.tail(top_n)
        
        # MAE comparison
        combined_mae = pd.concat([
            top_skus[['sku_id', 'mae']].assign(category='Best'),
            bottom_skus[['sku_id', 'mae']].assign(category='Worst')
        ])
        
        sns.barplot(data=combined_mae, x='mae', y='sku_id', hue='category', ax=axes[0, 0])
        axes[0, 0].set_title(f'Top/Bottom {top_n} SKUs by MAE')
        axes[0, 0].set_xlabel('Mean Absolute Error')
        
        # R² vs Service Level scatter
        scatter = axes[0, 1].scatter(sku_df['r2'], sku_df['service_level'], 
                                   c=sku_df['total_cost'], cmap='viridis', alpha=0.7)
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_ylabel('Service Level')
        axes[0, 1].set_title('R² vs Service Level (Color = Total Cost)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Total Cost')
        
        # Add target lines
        axes[0, 1].axhline(y=self.eval_config['service_level_threshold'], 
                          color='r', linestyle='--', alpha=0.7, label='Service Target')
        axes[0, 1].axvline(x=0.7, color='orange', linestyle='--', alpha=0.7, label='R² Target')
        axes[0, 1].legend()
        
        # Cost vs Demand relationship
        axes[1, 0].scatter(sku_df['avg_demand'], sku_df['total_cost'], alpha=0.7, 
                          color=self.colors['primary'])
        axes[1, 0].set_xlabel('Average Demand')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].set_title('Cost vs Demand Relationship')
        
        # Add trend line
        z = np.polyfit(sku_df['avg_demand'], sku_df['total_cost'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(sku_df['avg_demand'], p(sku_df['avg_demand']), "r--", alpha=0.8)
        
        # Performance distribution
        performance_metrics = ['mae', 'r2', 'service_level']
        performance_data = sku_df[performance_metrics]
        
        # Normalize for comparison (0-1 scale)
        performance_normalized = performance_data.copy()
        performance_normalized['mae'] = 1 - (performance_normalized['mae'] - performance_normalized['mae'].min()) / \
                                      (performance_normalized['mae'].max() - performance_normalized['mae'].min())
        
        performance_normalized.boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Performance Metrics Distribution')
        axes[1, 1].set_ylabel('Normalized Score (0-1)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SKU performance comparison plot saved to {save_path}")
        
        return fig
    
    def plot_time_series_analysis(self, df: pd.DataFrame, predictions: np.ndarray,
                                 sample_skus: Optional[List[str]] = None,
                                 title: str = "Time Series Analysis",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze time series patterns for selected SKUs.
        
        Args:
            df: DataFrame with time series data
            predictions: Model predictions
            sample_skus: List of SKU IDs to analyze (random sample if None)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        df_viz = df.copy()
        df_viz['predictions'] = predictions
        
        # Select sample SKUs if not provided
        if sample_skus is None:
            sample_skus = df_viz['sku_id'].unique()[:self.eval_config['visualization_skus']]
        
        n_skus = len(sample_skus)
        fig, axes = plt.subplots(n_skus, 1, figsize=(15, 4*n_skus))
        if n_skus == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, sku_id in enumerate(sample_skus):
            sku_data = df_viz[df_viz['sku_id'] == sku_id].sort_values('week')
            
            # Plot demand, inventory, and predictions
            ax = axes[i]
            
            # Demand and predictions
            ax.plot(sku_data['week'], sku_data['actual_demand'], 
                   label='Actual Demand', color=self.colors['primary'], linewidth=2)
            ax.plot(sku_data['week'], sku_data['target_quantity'], 
                   label='Target Quantity', color=self.colors['secondary'], linewidth=2)
            ax.plot(sku_data['week'], sku_data['predictions'], 
                   label='Predicted Quantity', color=self.colors['accent'], 
                   linewidth=2, linestyle='--')
            
            # Inventory level
            ax2 = ax.twinx()
            ax2.plot(sku_data['week'], sku_data['current_inventory'], 
                    label='Inventory Level', color=self.colors['success'], alpha=0.7)
            ax2.set_ylabel('Inventory Level', color=self.colors['success'])
            
            ax.set_xlabel('Week')
            ax.set_ylabel('Quantity')
            ax.set_title(f'{sku_id} - Time Series Analysis')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series analysis plot saved to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, evaluation_results: Dict, 
                                  df: pd.DataFrame, predictions: np.ndarray,
                                  output_dir: str = "visualizations") -> Dict[str, str]:
        """
        Create a comprehensive visualization report.
        
        Args:
            evaluation_results: Complete evaluation results
            df: Original DataFrame
            predictions: Model predictions
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of saved file paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        print("Creating comprehensive visualization report...")
        
        # 1. Prediction accuracy
        y_true = df['target_quantity'].values
        fig1 = self.plot_prediction_accuracy(
            y_true, predictions, 
            save_path=str(output_path / "prediction_accuracy.png")
        )
        saved_files['prediction_accuracy'] = str(output_path / "prediction_accuracy.png")
        plt.close(fig1)
        
        # 2. Cost breakdown
        fig2 = self.plot_cost_breakdown(
            evaluation_results['cost_metrics'],
            save_path=str(output_path / "cost_breakdown.png")
        )
        saved_files['cost_breakdown'] = str(output_path / "cost_breakdown.png")
        plt.close(fig2)
        
        # 3. Service level analysis
        fig3 = self.plot_service_level_analysis(
            evaluation_results['service_metrics'],
            save_path=str(output_path / "service_level_analysis.png")
        )
        saved_files['service_level_analysis'] = str(output_path / "service_level_analysis.png")
        plt.close(fig3)
        
        # 4. SKU performance comparison
        fig4 = self.plot_sku_performance_comparison(
            evaluation_results['sku_analysis'],
            save_path=str(output_path / "sku_performance_comparison.png")
        )
        saved_files['sku_performance_comparison'] = str(output_path / "sku_performance_comparison.png")
        plt.close(fig4)
        
        # 5. Time series analysis
        fig5 = self.plot_time_series_analysis(
            df, predictions,
            save_path=str(output_path / "time_series_analysis.png")
        )
        saved_files['time_series_analysis'] = str(output_path / "time_series_analysis.png")
        plt.close(fig5)
        
        print(f"Comprehensive visualization report created in {output_dir}/")
        print(f"Generated {len(saved_files)} visualization files")
        
        return saved_files

def create_model_comparison_plot(comparison_results: Dict, 
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization comparing multiple models.
    
    Args:
        comparison_results: Results from compare_models function
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    models = comparison_results['models']
    metrics_comparison = comparison_results['metrics_comparison']
    
    # Key metrics to compare
    key_metrics = [
        ('prediction_metrics_mae', 'MAE', 'lower_better'),
        ('prediction_metrics_r2', 'R²', 'higher_better'),
        ('service_metrics_avg_service_level', 'Service Level', 'higher_better'),
        ('cost_metrics_total_cost', 'Total Cost', 'lower_better')
    ]
    
    for i, (metric_key, metric_name, direction) in enumerate(key_metrics):
        ax = axes[i // 2, i % 2]
        
        if metric_key in metrics_comparison:
            values = [metrics_comparison[metric_key][model] for model in models]
            
            # Color bars based on performance
            if direction == 'lower_better':
                colors = ['green' if v == min(values) else 'red' if v == max(values) else 'blue' 
                         for v in values]
            else:
                colors = ['green' if v == max(values) else 'red' if v == min(values) else 'blue' 
                         for v in values]
            
            bars = ax.bar(models, values, color=colors, alpha=0.7)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    return fig
