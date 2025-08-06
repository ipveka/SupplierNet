"""
Comprehensive evaluation script for SupplierNet models.
Combines testing, evaluation metrics, and visualization generation.

=== EVALUATION SPECIFICATION ===

MODEL EVALUATION TARGET:
- Evaluates predictions of 'target_quantity': Optimal supply order quantities
- Assesses model's ability to predict cost-optimal procurement/manufacturing decisions
- Compares predicted vs actual optimal quantities across multiple SKUs and time periods

COMPREHENSIVE METRICS CALCULATED:

1. PREDICTION ACCURACY:
   - MAE: Mean Absolute Error (average units deviation from optimal)
   - RMSE: Root Mean Square Error (emphasizes large prediction errors)
   - R²: Coefficient of determination (proportion of variance explained)
   - MAPE: Mean Absolute Percentage Error (relative prediction accuracy)

2. BUSINESS IMPACT METRICS:
   - Total Cost Analysis: Holding, shortage, ordering, production costs
   - Service Level Performance: Stockout frequency, perfect service rate
   - SKU-Level Performance: Individual product prediction quality
   - Cost Efficiency: Predicted vs optimal cost comparison

3. OPERATIONAL METRICS:
   - Inventory Turnover: How well model manages inventory levels
   - Lead Time Utilization: Efficiency in lead time management
   - Demand Fulfillment: Percentage of demand met without stockouts

VISUALIZATIONS GENERATED:
- Prediction accuracy scatter plots and residual analysis
- Cost breakdown charts (pie charts and bar graphs)
- Service level performance gauges and distributions
- SKU performance heatmaps and time series comparisons
- Business impact summary with executive dashboard
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
import json
from typing import Dict, Optional
from pathlib import Path

# Local imports
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import get_config, set_random_seeds
from data.simulation import SupplyChainSimulator
from utils.data_processing import DataProcessor
from utils.evaluation import SupplyChainEvaluator
from utils.visualization import SupplyChainVisualizer
from models.neural_net import load_model
from src.test import load_test_data, prepare_test_data, run_inference

def comprehensive_model_evaluation(model_path: str, preprocessor_path: str,
                                   test_data_path: str = 'data/test_dataset.csv',
                                   output_dir: str = 'evaluation_results',
                                   generate_new_test_data: bool = False) -> Dict:
    """
    Complete model evaluation pipeline with metrics and visualizations.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to fitted preprocessor
        test_data_path: Path to test dataset
        output_dir: Directory for evaluation outputs
        generate_new_test_data: Whether to generate new test data
        
    Returns:
        Complete evaluation results
    """
    print("=" * 70)
    print("SupplierNet Comprehensive Model Evaluation")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    print(f"\nLoading model from {model_path}")
    model, config = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Load preprocessor
    print(f"Loading preprocessor from {preprocessor_path}")
    processor = DataProcessor(config)
    processor.load_preprocessor(preprocessor_path)
    
    # Set random seeds
    set_random_seeds(config['random_seed'])
    
    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data(config, test_data_path, generate_new_test_data)
    
    # Prepare test data
    test_data = prepare_test_data(test_df, processor)
    
    # Run inference
    print("\nRunning model inference...")
    predictions = run_inference(model, test_data['X_test'], device)
    
    # Inverse transform predictions if needed
    if processor.target_scaler is not None:
        predictions = processor.inverse_transform_targets(predictions)
        y_test = processor.inverse_transform_targets(test_data['y_test'])
        test_data['y_test'] = y_test
    
    # Initialize evaluator
    print("\nInitializing comprehensive evaluation...")
    evaluator = SupplyChainEvaluator(config)
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_model_performance(
        test_data['raw_data'], predictions
    )
    
    # Initialize visualizer
    print("Creating visualizations...")
    visualizer = SupplyChainVisualizer(config)
    
    # Create visualization directory
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive visualization report
    visualization_files = visualizer.create_comprehensive_report(
        evaluation_results, test_data['raw_data'], predictions, str(viz_dir)
    )
    
    # Compile final results
    final_results = {
        'model_info': {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'test_data_path': test_data_path,
            'config': config
        },
        'data_info': {
            'n_test_samples': len(predictions),
            'n_skus': test_data['raw_data']['sku_id'].nunique(),
            'time_horizon': test_data['raw_data']['week'].max() + 1,
            'feature_names': test_data['feature_names']
        },
        'evaluation_results': evaluation_results,
        'visualization_files': visualization_files,
        'predictions_sample': {
            'true_values': test_data['y_test'][:50].tolist(),
            'predictions': predictions[:50].tolist()
        }
    }
    
    # Save complete results
    results_file = output_path / "comprehensive_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Create executive summary
    create_executive_summary(final_results, output_path)
    
    # Print summary
    print_evaluation_summary(final_results)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}/")
    
    return final_results

def create_executive_summary(results: Dict, output_path: Path) -> None:
    """Create an executive summary report."""
    eval_results = results['evaluation_results']
    summary = eval_results['evaluation_summary']
    business_impact = eval_results['business_impact']
    
    summary_text = f"""
        # SupplierNet Model Evaluation - Executive Summary

        ## Model Performance Overview
        - **Overall Performance Score**: {business_impact['overall_performance_score']:.3f}/1.000
        - **Test Dataset**: {results['data_info']['n_test_samples']:,} samples across {results['data_info']['n_skus']} SKUs
        - **Time Horizon**: {results['data_info']['time_horizon']} weeks

        ## Key Performance Metrics

        ### Prediction Accuracy
        - **Mean Absolute Error (MAE)**: {summary['model_accuracy']['mae']:.4f}
        - **Root Mean Square Error (RMSE)**: {summary['model_accuracy']['rmse']:.4f}
        - **R-squared (R²)**: {summary['model_accuracy']['r2']:.4f}
        - **Mean Absolute Percentage Error (MAPE)**: {summary['model_accuracy']['mape']:.2f}%

        ### Cost Performance
        - **Total Cost**: ${summary['cost_performance']['total_cost']:,.2f}
        - **Average Cost per Period**: ${summary['cost_performance']['avg_cost_per_period']:.2f}
        - **Cost Breakdown**:
        - Holding Costs: {summary['cost_performance']['cost_breakdown']['holding']:.1%}
        - Shortage Costs: {summary['cost_performance']['cost_breakdown']['shortage']:.1%}
        - Ordering Costs: {summary['cost_performance']['cost_breakdown']['ordering']:.1%}
        - Production Costs: {summary['cost_performance']['cost_breakdown']['production']:.1%}

        ### Service Level Performance
        - **Average Service Level**: {summary['service_performance']['avg_service_level']:.1%}
        - **Target Service Level**: {results['evaluation_results']['config']['evaluation']['service_level_threshold']:.1%}
        - **Service Level Achievement**: {summary['service_performance']['avg_service_level']/results['evaluation_results']['config']['evaluation']['service_level_threshold']:.1%}
        - **Stockout Frequency**: {summary['service_performance']['stockout_frequency']:.1%}
        - **Perfect Service Rate**: {summary['service_performance']['perfect_service_rate']:.1%}

        ### SKU-Level Performance
        - **Total SKUs Analyzed**: {summary['sku_performance']['total_skus']}
        - **SKUs with Good Predictions (R²>0.7)**: {summary['sku_performance']['skus_with_good_predictions']}
        - **SKUs with High Service Level**: {summary['sku_performance']['skus_with_high_service']}
        - **Average Performance Across SKUs**:
        - MAE: {summary['sku_performance']['avg_performance_across_skus']['mae']:.4f}
        - R²: {summary['sku_performance']['avg_performance_across_skus']['r2']:.4f}
        - Service Level: {summary['sku_performance']['avg_performance_across_skus']['service_level']:.1%}

        ## Business Impact Assessment

        ### Performance Scores
        - **Accuracy Score**: {business_impact['accuracy_score']:.3f}/1.000
        - **Cost Efficiency Score**: {business_impact['cost_efficiency_score']:.3f}/1.000
        - **Service Level Score**: {business_impact['service_level_score']:.3f}/1.000

        ### Key Insights
        - **Model Reliability**: {'High' if business_impact['accuracy_score'] > 0.8 else 'Medium' if business_impact['accuracy_score'] > 0.6 else 'Low'}
        - **Cost Management**: {'Excellent' if business_impact['cost_efficiency_score'] > 0.9 else 'Good' if business_impact['cost_efficiency_score'] > 0.7 else 'Needs Improvement'}
        - **Service Quality**: {'Excellent' if business_impact['service_level_score'] > 0.95 else 'Good' if business_impact['service_level_score'] > 0.9 else 'Needs Improvement'}

        ## Recommendations

        ### Strengths
        - {f"Strong prediction accuracy with R² of {summary['model_accuracy']['r2']:.3f}" if summary['model_accuracy']['r2'] > 0.7 else ""}
        - {f"Good service level achievement at {summary['service_performance']['avg_service_level']:.1%}" if summary['service_performance']['avg_service_level'] > 0.9 else ""}
        - {f"Effective cost management with low shortage costs" if summary['cost_performance']['cost_breakdown']['shortage'] < 0.2 else ""}

        ### Areas for Improvement
        - {f"Prediction accuracy could be improved (R² = {summary['model_accuracy']['r2']:.3f})" if summary['model_accuracy']['r2'] < 0.7 else ""}
        - {f"Service level below target ({summary['service_performance']['avg_service_level']:.1%} vs target)" if summary['service_performance']['avg_service_level'] < results['evaluation_results']['config']['evaluation']['service_level_threshold'] else ""}
        - {f"High shortage costs ({summary['cost_performance']['cost_breakdown']['shortage']:.1%} of total)" if summary['cost_performance']['cost_breakdown']['shortage'] > 0.3 else ""}

        ### Next Steps
        1. **Model Optimization**: Consider hyperparameter tuning or architecture improvements
        2. **Feature Engineering**: Explore additional features that could improve prediction accuracy
        3. **Cost Optimization**: Review ordering policies to reduce shortage and holding costs
        4. **SKU Segmentation**: Develop specialized models for different SKU categories
        5. **Real-world Validation**: Test model performance on actual historical data

        ---
        *Report generated by SupplierNet Evaluation System*
    """
    
    # Save executive summary
    summary_file = output_path / "executive_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(f"Executive summary saved to: {summary_file}")

def print_evaluation_summary(results: Dict) -> None:
    """Print evaluation summary to console."""
    eval_results = results['evaluation_results']
    summary = eval_results['evaluation_summary']
    business_impact = eval_results['business_impact']
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 DATASET INFO:")
    print(f"  • Test samples: {results['data_info']['n_test_samples']:,}")
    print(f"  • SKUs analyzed: {results['data_info']['n_skus']}")
    print(f"  • Time horizon: {results['data_info']['time_horizon']} weeks")
    
    print(f"\n🎯 PREDICTION ACCURACY:")
    print(f"  • MAE: {summary['model_accuracy']['mae']:.4f}")
    print(f"  • RMSE: {summary['model_accuracy']['rmse']:.4f}")
    print(f"  • R²: {summary['model_accuracy']['r2']:.4f}")
    print(f"  • MAPE: {summary['model_accuracy']['mape']:.2f}%")
    
    print(f"\n💰 COST PERFORMANCE:")
    print(f"  • Total cost: ${summary['cost_performance']['total_cost']:,.2f}")
    print(f"  • Avg cost/period: ${summary['cost_performance']['avg_cost_per_period']:.2f}")
    print(f"  • Shortage costs: {summary['cost_performance']['cost_breakdown']['shortage']:.1%}")
    print(f"  • Holding costs: {summary['cost_performance']['cost_breakdown']['holding']:.1%}")
    
    print(f"\n🎯 SERVICE LEVEL:")
    print(f"  • Average: {summary['service_performance']['avg_service_level']:.1%}")
    print(f"  • Target: {results['evaluation_results']['config']['evaluation']['service_level_threshold']:.1%}")
    print(f"  • Stockout frequency: {summary['service_performance']['stockout_frequency']:.1%}")
    print(f"  • Perfect service rate: {summary['service_performance']['perfect_service_rate']:.1%}")
    
    print(f"\n📈 BUSINESS IMPACT:")
    print(f"  • Overall score: {business_impact['overall_performance_score']:.3f}/1.000")
    print(f"  • Accuracy score: {business_impact['accuracy_score']:.3f}/1.000")
    print(f"  • Cost efficiency: {business_impact['cost_efficiency_score']:.3f}/1.000")
    print(f"  • Service quality: {business_impact['service_level_score']:.3f}/1.000")
    
    print(f"\n🏆 SKU PERFORMANCE:")
    print(f"  • SKUs with good predictions: {summary['sku_performance']['skus_with_good_predictions']}/{summary['sku_performance']['total_skus']}")
    print(f"  • SKUs with high service: {summary['sku_performance']['skus_with_high_service']}/{summary['sku_performance']['total_skus']}")
    
    # Performance assessment
    overall_score = business_impact['overall_performance_score']
    if overall_score > 0.8:
        assessment = "🟢 EXCELLENT"
    elif overall_score > 0.6:
        assessment = "🟡 GOOD"
    else:
        assessment = "🔴 NEEDS IMPROVEMENT"
    
    print(f"\n🎯 OVERALL ASSESSMENT: {assessment}")
    print("=" * 70)

def main():
    """Main evaluation script with command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive SupplierNet model evaluation')
    parser.add_argument('--model-path', type=str, default='../models/best_model.pth',
                       help='Path to trained model (default: ../models/best_model.pth)')
    parser.add_argument('--preprocessor-path', type=str,
                       help='Path to fitted preprocessor (auto-detected if not provided)')
    parser.add_argument('--test-data-path', type=str, default='data/test_dataset.csv',
                       help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory for evaluation outputs')
    parser.add_argument('--generate-new-test-data', action='store_true',
                       help='Generate new test data')
    
    args = parser.parse_args()
    
    # Auto-detect preprocessor path if not provided
    if args.preprocessor_path is None:
        args.preprocessor_path = args.model_path.replace('.pth', '_preprocessor.pkl')
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.preprocessor_path):
        print(f"Error: Preprocessor file not found: {args.preprocessor_path}")
        return
    
    try:
        results = comprehensive_model_evaluation(
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path,
            test_data_path=args.test_data_path,
            output_dir=args.output_dir,
            generate_new_test_data=args.generate_new_test_data
        )
        
        print(f"\n✅ Comprehensive evaluation completed successfully!")
        print(f"📁 Results available in: {args.output_dir}/")
        
    except Exception as e:
        print(f"ERROR: Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
