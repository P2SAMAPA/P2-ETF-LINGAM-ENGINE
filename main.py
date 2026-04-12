"""
P2-ETF-LINGAM-Engine Main Training Script
=========================================
Main entry point for running the causal trading engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data.loader import load_etf_data, get_universe_data, split_data, get_shrinking_windows
from data.preprocessing import handle_missing_values, remove_outliers, normalize_features
from core.lingam_engine import LingamEngine
from core.causal_analyzer import CausalAnalyzer
from core.metrics import calculate_all_metrics, calculate_consensus_score
from core.consensus import ConsensusScorer
from modules.fi_commodity.causal_discovery import FICausalDiscovery
from modules.fi_commodity.leader_identifier import FILEaderIdentifier
from modules.fi_commodity.signal_generator import FISignalGenerator
from modules.equity.causal_discovery import EquityCausalDiscovery
from modules.equity.leader_identifier import EquityLeaderIdentifier
from modules.equity.signal_generator import EquitySignalGenerator
from output.predictions import PredictionFormatter
from output.hf_uploader import HFUploader


def run_fixed_split_training(universe: str, use_bootstrap: bool = True):
    """
    Run training with fixed 80/10/10 split.

    Args:
        universe: 'fi_commodity' or 'equity'
        use_bootstrap: Whether to use bootstrap for confidence

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Running FIXED SPLIT training for {universe}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    returns = get_universe_data(universe)

    # Split data
    print("Splitting data (80/10/10)...")
    train, val, test = split_data(returns)

    print(f"  Train: {len(train)} samples")
    print(f"  Val: {len(val)} samples")
    print(f"  Test: {len(test)} samples")

    # Select causal discovery module
    if universe == 'fi_commodity':
        causal_discovery = FICausalDiscovery()
        leader_identifier = FILEaderIdentifier()
        signal_generator = FISignalGenerator()
        assets = config.FI_COMMODITY_ASSETS
        benchmark = config.FI_COMMODITY_BENCHMARK
    else:
        causal_discovery = EquityCausalDiscovery()
        leader_identifier = EquityLeaderIdentifier()
        signal_generator = EquitySignalGenerator()
        assets = config.EQUITY_ASSETS
        benchmark = config.EQUITY_BENCHMARK

    # Prepare and run causal discovery
    print("\nRunning causal discovery...")
    data = causal_discovery.prepare_data(train, val)
    causal_results = causal_discovery.discover_causal_structure(data, use_bootstrap)

    print(f"  Identified leader: {causal_results['leader']}")
    print(f"  Causal edges found: {len(causal_results['causal_edges'])}")

    # Get leader predictions
    print("\nGenerating leader predictions...")
    predictions = causal_discovery.get_leader_predictions()

    print("  Top leaders by causal strength:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"    {i}. {pred['ticker']}: {pred['causal_influence']:.4f} (confidence: {pred['confidence']:.2%})")

    # Generate leader report
    print("\nGenerating leader report...")
    returns_with_benchmark = pd.concat([returns, returns[benchmark]], axis=1) if benchmark in returns.columns else returns
    leader_report = leader_identifier.generate_leader_report(predictions, [], returns)

    print(f"  Consensus leader: {leader_report['consensus_leader']}")
    print(f"  Conviction: {leader_report['consensus_conviction']:.2%}")

    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    metrics = {}
    if leader_report['consensus_leader'] in returns.columns:
        leader_returns = returns[leader_report['consensus_leader']].dropna()
        metrics = calculate_all_metrics(leader_returns)
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Generate signals
    print("\nGenerating trading signals...")
    signals = signal_generator.generate_signals(leader_report, returns, config.PREDICTION_DATE)

    # Handle case where signals is None or missing expected keys
    if signals is None:
        print("  Warning: No signals generated. Using default empty signal.")
        signals = {
            'primary_signal': {'ticker': 'N/A'},
            'confidence': 0.0,
            'signals_list': []
        }
    elif 'primary_signal' not in signals or signals['primary_signal'] is None:
        print("  Warning: Primary signal missing. Using fallback.")
        signals['primary_signal'] = {'ticker': 'N/A'}
        signals['confidence'] = 0.0

    print(f"  Primary signal: {signals['primary_signal']['ticker']}")
    print(f"  Confidence: {signals['confidence']:.1f}%")

    return {
        'universe': universe,
        'training_mode': 'fixed',
        'causal_results': causal_results,
        'leader_report': leader_report,
        'signals': signals,
        'metrics': metrics,
        'train_period': f"{train.index[0]} to {train.index[-1]}",
        'test_period': f"{test.index[0]} to {test.index[-1]}"
    }


def run_shrinking_window_training(universe: str):
    """
    Run training with shrinking windows and consensus scoring.

    Args:
        universe: 'fi_commodity' or 'equity'

    Returns:
        Dictionary with training results (compatible with fixed split)
    """
    print(f"\n{'='*60}")
    print(f"Running SHRINKING WINDOW training for {universe}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    returns = get_universe_data(universe)

    # Get shrinking window configuration
    start_years = config.SHRINKING_WINDOW_YEARS
    end_date = config.DATA_END_DATE

    # Initialize consensus scorer
    scorer = ConsensusScorer(
        weights=config.CONSENSUS_WEIGHTS,
        exclude_negative_returns=config.NEGATIVE_RETURN_ZERO_WEIGHT
    )

    # Generate windows
    print(f"Generating {len(start_years)} shrinking windows...")
    windows = scorer.generate_shrinking_window_results(
        returns,
        config.FI_COMMODITY_ASSETS if universe == 'fi_commodity' else config.EQUITY_ASSETS,
        start_years,
        end_date
    )

    print(f"  Created {len(windows)} valid windows")

    # Run causal discovery on each window
    print("\nRunning causal discovery on each window...")
    window_results = []

    for i, window in enumerate(windows):
        print(f"\n  Window {i+1}/{len(windows)}: {window['window_start']} to {window['window_end']}")

        # Select module
        if universe == 'fi_commodity':
            causal_discovery = FICausalDiscovery()
        else:
            causal_discovery = EquityCausalDiscovery()

        # Prepare and run
        data = causal_discovery.prepare_data(window['returns'])
        causal_results = causal_discovery.discover_causal_structure(data, use_bootstrap=False)

        if causal_results['leader']:
            # Calculate consensus score for leader
            if causal_results['leader'] in window['returns'].columns:
                leader_returns = window['returns'][causal_results['leader']].dropna()
                if len(leader_returns) >= 20:
                    window_score = scorer.calculate_window_score(leader_returns)

                    # Store the window data including returns for consensus scoring
                    window_results.append({
                        'window_start': window['window_start'],
                        'window_end': window['window_end'],
                        'leader_ticker': causal_results['leader'],
                        'leader_score': causal_results['leader_score'],
                        'consensus_score': window_score,
                        'causal_results': causal_results,
                        'returns': window['returns']  # Add returns for consensus calculation
                    })

                    print(f"    Leader: {causal_results['leader']}, Score: {window_score:.4f}")

    # Get consensus leader
    print("\n" + "="*40)
    print("CONSENSUS RESULTS")
    print("="*40)

    if window_results:
        # Use consensus scorer to get final leader
        consensus_df = scorer.calculate_consensus_scores(
            window_results,
            config.FI_COMMODITY_ASSETS if universe == 'fi_commodity' else config.EQUITY_ASSETS
        )

        final_leader, conviction, top_3 = scorer.get_final_leader(consensus_df)

        print(f"\nFinal Leader: {final_leader}")
        print(f"Conviction: {conviction:.2%}")
        print(f"\nTop 3 Picks:")
        for i, pick in enumerate(top_3, 1):
            print(f"  {i}. {pick['ticker']}: score={pick['score']:.4f}, ann_return={pick['ann_return']:.2%}")

        # Build metrics for the final leader (use its average annualized return as proxy)
        metrics = {}
        if final_leader:
            # Find the leader's average metrics from top_3
            leader_metrics = next((p for p in top_3 if p['ticker'] == final_leader), None)
            if leader_metrics:
                metrics = {
                    'total_return': leader_metrics.get('ann_return', 0.0),
                    'sharpe_ratio': leader_metrics.get('sharpe', 0.0),
                    'max_drawdown': leader_metrics.get('max_dd', 0.0),
                    'win_rate': 0.0,  # Not available in consensus
                    'best_day': 0.0,
                }
            else:
                metrics = {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 'best_day': 0.0}

        # Build signals dict similar to fixed split
        signals = {
            'primary_signal': {'ticker': final_leader, 'ann_return': metrics.get('total_return', 0.0)},
            'confidence': conviction,
            'all_signals': top_3,
            'universe': universe,
            'date': config.PREDICTION_DATE,
        }

        # Build causal_results dict (mock structure for upload compatibility)
        causal_results = {
            'leader': final_leader,
            'followers': [(p['ticker'], p['score']) for p in top_3[1:]] if len(top_3) > 1 else [],
            'causal_edges': [],  # Not used for shrinking window
        }

        return {
            'universe': universe,
            'training_mode': 'shrinking',
            'final_leader': final_leader,
            'conviction': conviction,
            'top_3_picks': top_3,
            'window_results': window_results,
            'n_windows': len(window_results),
            'signals': signals,
            'metrics': metrics,
            'causal_results': causal_results,
        }
    else:
        print("\nNo valid window results found.")
        return {
            'universe': universe,
            'training_mode': 'shrinking',
            'final_leader': None,
            'conviction': 0,
            'top_3_picks': [],
            'window_results': [],
            'n_windows': 0,
            'signals': None,
            'metrics': {},
            'causal_results': {},
        }


def run_full_pipeline():
    """
    Run the complete training pipeline for both universes.
    """
    print("\n" + "="*80)
    print("P2-ETF-LINGAM-ENGINE - FULL TRAINING PIPELINE")
    print("="*80 + "\n")

    all_results = {}

    # Run for FI/Commodity
    print("\n" + "#"*60)
    print("# FIXED INCOME / COMMODITY MODULE")
    print("#"*60)

    fi_fixed = run_fixed_split_training('fi_commodity')
    fi_shrinking = run_shrinking_window_training('fi_commodity')

    all_results['fi_commodity'] = {
        'fixed': fi_fixed,
        'shrinking': fi_shrinking
    }

    # Run for Equity
    print("\n" + "#"*60)
    print("# EQUITY MODULE")
    print("#"*60)

    eq_fixed = run_fixed_split_training('equity')
    eq_shrinking = run_shrinking_window_training('equity')

    all_results['equity'] = {
        'fixed': eq_fixed,
        'shrinking': eq_shrinking
    }

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    print("\nFI/COMMODITY MODULE:")
    print(f"  Fixed Split Leader: {fi_fixed.get('leader_report', {}).get('consensus_leader', 'N/A')}")
    print(f"  Shrinking Window Leader: {fi_shrinking.get('final_leader', 'N/A')}")

    print("\nEQUITY MODULE:")
    print(f"  Fixed Split Leader: {eq_fixed.get('leader_report', {}).get('consensus_leader', 'N/A')}")
    print(f"  Shrinking Window Leader: {eq_shrinking.get('final_leader', 'N/A')}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='P2-ETF-LINGAM-Engine Training Script'
    )

    parser.add_argument(
        '--universe',
        type=str,
        choices=['fi_commodity', 'equity', 'all'],
        default='all',
        help='Universe to train on'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['fixed', 'shrinking', 'both'],
        default='both',
        help='Training mode'
    )

    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help='Use bootstrap for confidence estimation'
    )

    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload results to HuggingFace'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        help='Save results to file'
    )

    args = parser.parse_args()

    # Run training
    results = {}

    if args.universe in ['fi_commodity', 'all']:
        if args.mode in ['fixed', 'both']:
            results['fi_commodity_fixed'] = run_fixed_split_training('fi_commodity', args.bootstrap)
        if args.mode in ['shrinking', 'both']:
            results['fi_commodity_shrinking'] = run_shrinking_window_training('fi_commodity')

    if args.universe in ['equity', 'all']:
        if args.mode in ['fixed', 'both']:
            results['equity_fixed'] = run_fixed_split_training('equity', args.bootstrap)
        if args.mode in ['shrinking', 'both']:
            results['equity_shrinking'] = run_shrinking_window_training('equity')

    # Save to file if requested
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Upload to HuggingFace if requested
    if args.upload:
        print("\nUploading results to HuggingFace...")
        uploader = HFUploader()
        predictions = []

        for key, result in results.items():
            # Both fixed and shrinking results now have 'signals', 'metrics', 'causal_results', 'training_mode'
            if result and result.get('signals') and result['signals'].get('primary_signal'):
                formatter = PredictionFormatter()
                pred = formatter.format_prediction(
                    result['signals'],
                    result.get('metrics', {}),
                    result.get('causal_results', {}),
                    result.get('training_mode', 'unknown')
                )
                # Ensure universe is set in the prediction
                if 'universe' not in pred and 'universe' in result:
                    pred['universe'] = result['universe']
                predictions.append(pred)

        if predictions:
            uploader.upload_predictions(predictions)

    print("\nTraining complete!")

    return results


if __name__ == "__main__":
    results = main()
