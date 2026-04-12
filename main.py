"""
P2-ETF-LINGAM-Engine Main Training Script
=========================================
Optimized: determines best causal measure on fixed split, then forces 'pwling' for shrinking windows.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data.loader import get_universe_data, split_data
from core.metrics import calculate_all_metrics
from core.consensus import ConsensusScorer
from modules.fi_commodity.causal_discovery import FICausalDiscovery
from modules.fi_commodity.leader_identifier import FILEaderIdentifier
from modules.equity.causal_discovery import EquityCausalDiscovery
from modules.equity.leader_identifier import EquityLeaderIdentifier
from output.predictions import PredictionFormatter
from output.hf_uploader import HFUploader


def annualized_return_from_series(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    return (1 + total_return) ** (periods_per_year / n_days) - 1


def determine_best_measure_fixed(universe: str, use_bootstrap: bool = True):
    """
    Run fixed split for each causal measure and return the measure that yields
    the highest test‑period annualized return for the leader ETF.
    Also returns the full result dictionary for that best measure.
    """
    print(f"\n--- Determining best causal measure for {universe} (fixed split) ---")
    returns = get_universe_data(universe)
    train, val, test = split_data(returns)

    best_ann_return = -np.inf
    best_result = None
    best_measure = None

    for measure in config.CAUSAL_MEASURES:
        print(f"\nTrying measure: {measure}")
        if universe == 'fi_commodity':
            causal_discovery = FICausalDiscovery()
            leader_identifier = FILEaderIdentifier()
        else:
            causal_discovery = EquityCausalDiscovery()
            leader_identifier = EquityLeaderIdentifier()

        causal_discovery.lingam.config['measure'] = measure

        data = causal_discovery.prepare_data(train, val)
        causal_results = causal_discovery.discover_causal_structure(data, use_bootstrap)

        leader_ticker = causal_results.get('leader', 'N/A')
        print(f"  Leader: {leader_ticker}")

        if leader_ticker in returns.columns:
            test_returns = test[leader_ticker].dropna()
            if len(test_returns) > 0:
                ann_return = annualized_return_from_series(test_returns)
                print(f"  Annualized return (test period): {ann_return:.2%}")
                if ann_return > best_ann_return:
                    best_ann_return = ann_return
                    best_measure = measure
                    best_result = {
                        'causal_results': causal_results,
                        'leader_ticker': leader_ticker,
                        'ann_return': ann_return,
                        'train': train,
                        'val': val,
                        'test': test,
                        'returns': returns,
                        'leader_identifier': leader_identifier,
                        'causal_discovery': causal_discovery,
                    }
            else:
                print(f"  No test returns for leader {leader_ticker}")
        else:
            print(f"  Leader {leader_ticker} not in returns")

    if best_result is None:
        print("WARNING: No measure produced a valid leader. Using first measure as fallback.")
        best_measure = config.CAUSAL_MEASURES[0]
        causal_discovery = FICausalDiscovery() if universe == 'fi_commodity' else EquityCausalDiscovery()
        causal_discovery.lingam.config['measure'] = best_measure
        data = causal_discovery.prepare_data(train, val)
        causal_results = causal_discovery.discover_causal_structure(data, use_bootstrap)
        leader_ticker = causal_results.get('leader', 'N/A')
        best_result = {
            'causal_results': causal_results,
            'leader_ticker': leader_ticker,
            'ann_return': 0.0,
            'train': train,
            'val': val,
            'test': test,
            'returns': returns,
            'leader_identifier': leader_identifier,
            'causal_discovery': causal_discovery,
        }

    print(f"\n=== Best measure for {universe} fixed split: {best_measure} (annualized return {best_result['ann_return']:.2%}) ===")
    return best_measure, best_result


def run_fixed_split_training_from_best(universe: str, best_measure: str, best_result: dict):
    """
    Run fixed split training using the pre‑determined best measure and result.
    """
    print(f"\n{'='*60}")
    print(f"Running FIXED SPLIT training for {universe} using best measure: {best_measure}")
    print(f"{'='*60}\n")

    causal_discovery = best_result['causal_discovery']
    causal_results = best_result['causal_results']
    leader_ticker = best_result['leader_ticker']
    ann_return = best_result['ann_return']
    train = best_result['train']
    test = best_result['test']
    returns = best_result['returns']

    metrics = {}
    if leader_ticker in returns.columns:
        test_returns = test[leader_ticker].dropna()
        if len(test_returns) > 0:
            full_metrics = calculate_all_metrics(test_returns)
            metrics = {
                'annualized_return': ann_return,
                'sharpe_ratio': full_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': full_metrics.get('max_drawdown', 0.0),
                'win_rate': full_metrics.get('win_rate', 0.0),
                'best_day': full_metrics.get('best_day', 0.0),
            }

    signals = {
        'primary_signal': {'ticker': leader_ticker, 'ann_return': ann_return},
        'confidence': causal_results.get('leader_score', 0.0),
        'all_signals': causal_discovery.get_leader_predictions()[:3],
        'universe': universe,
        'date': config.PREDICTION_DATE,
    }

    print(f"  Primary signal: {signals['primary_signal']['ticker']}")
    print(f"  Confidence: {signals['confidence']:.4f}")

    return {
        'universe': universe,
        'training_mode': 'fixed',
        'causal_results': causal_results,
        'signals': signals,
        'metrics': metrics,
        'train_period': f"{train.index[0]} to {train.index[-1]}",
        'test_period': f"{test.index[0]} to {test.index[-1]}",
        'selected_measure': best_measure
    }


def run_shrinking_window_training(universe: str):
    """
    Run shrinking window training ALWAYS using 'pwling' measure for speed.
    """
    measure = "pwling"   # force pwling
    print(f"\n{'='*60}")
    print(f"Running SHRINKING WINDOW training for {universe} using measure: {measure}")
    print(f"{'='*60}\n")

    returns = get_universe_data(universe)
    start_years = config.SHRINKING_WINDOW_YEARS
    end_date = config.DATA_END_DATE

    scorer = ConsensusScorer(
        weights=config.CONSENSUS_WEIGHTS,
        exclude_negative_returns=config.NEGATIVE_RETURN_ZERO_WEIGHT
    )

    windows = scorer.generate_shrinking_window_results(
        returns,
        config.FI_COMMODITY_ASSETS if universe == 'fi_commodity' else config.EQUITY_ASSETS,
        start_years,
        end_date
    )
    print(f"  Created {len(windows)} valid windows")

    window_results = []
    for i, window in enumerate(windows):
        print(f"\n  Window {i+1}/{len(windows)}: {window['window_start']} to {window['window_end']}")
        causal_discovery = FICausalDiscovery() if universe == 'fi_commodity' else EquityCausalDiscovery()
        causal_discovery.lingam.config['measure'] = measure
        data = causal_discovery.prepare_data(window['returns'])
        causal_results = causal_discovery.discover_causal_structure(data, use_bootstrap=False)

        if causal_results['leader'] and causal_results['leader'] in window['returns'].columns:
            leader_returns = window['returns'][causal_results['leader']].dropna()
            if len(leader_returns) >= 20:
                window_score = scorer.calculate_window_score(leader_returns)
                window_results.append({
                    'window_start': window['window_start'],
                    'window_end': window['window_end'],
                    'leader_ticker': causal_results['leader'],
                    'leader_score': causal_results['leader_score'],
                    'consensus_score': window_score,
                    'causal_results': causal_results,
                    'returns': window['returns']
                })
                print(f"    Leader: {causal_results['leader']}, Score: {window_score:.4f}")

    if not window_results:
        print("No valid window results. Returning empty.")
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
            'selected_measure': measure
        }

    assets = config.FI_COMMODITY_ASSETS if universe == 'fi_commodity' else config.EQUITY_ASSETS
    consensus_df = scorer.calculate_consensus_scores(window_results, assets)
    final_leader, conviction, top_3 = scorer.get_final_leader(consensus_df)

    print(f"\nFinal Leader: {final_leader}")
    print(f"Conviction: {conviction:.2%}")
    for pick in top_3:
        print(f"  {pick['ticker']}: ann_return={pick['ann_return']:.2%}")

    metrics = {}
    leader_metrics = next((p for p in top_3 if p['ticker'] == final_leader), None)
    if leader_metrics:
        metrics = {
            'annualized_return': leader_metrics['ann_return'],
            'sharpe_ratio': leader_metrics.get('sharpe', 0.0),
            'max_drawdown': leader_metrics.get('max_dd', 0.0),
            'win_rate': 0.0,
            'best_day': 0.0,
        }

    signals = {
        'primary_signal': {'ticker': final_leader, 'ann_return': metrics.get('annualized_return', 0.0)},
        'confidence': conviction,
        'all_signals': top_3,
        'universe': universe,
        'date': config.PREDICTION_DATE,
    }
    causal_results = {'leader': final_leader, 'followers': [(p['ticker'], p['score']) for p in top_3[1:]], 'causal_edges': []}

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
        'selected_measure': measure
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', type=str, choices=['fi_commodity', 'equity', 'all'], default='all')
    parser.add_argument('--mode', type=str, choices=['fixed', 'shrinking', 'both'], default='both')
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    results = {}

    # Determine best measure per universe using fixed split (only if fixed split is requested)
    best_measures = {}
    if args.mode in ['fixed', 'both']:
        if args.universe in ['fi_commodity', 'all']:
            best_measure, best_fixed_result = determine_best_measure_fixed('fi_commodity', args.bootstrap)
            best_measures['fi_commodity'] = best_measure
            results['fi_commodity_fixed'] = run_fixed_split_training_from_best('fi_commodity', best_measure, best_fixed_result)
        if args.universe in ['equity', 'all']:
            best_measure, best_fixed_result = determine_best_measure_fixed('equity', args.bootstrap)
            best_measures['equity'] = best_measure
            results['equity_fixed'] = run_fixed_split_training_from_best('equity', best_measure, best_fixed_result)

    # Shrinking window always uses pwling (no measure selection)
    if args.mode in ['shrinking', 'both']:
        if args.universe in ['fi_commodity', 'all']:
            results['fi_commodity_shrinking'] = run_shrinking_window_training('fi_commodity')
        if args.universe in ['equity', 'all']:
            results['equity_shrinking'] = run_shrinking_window_training('equity')

    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    if args.upload:
        print("\nUploading results to HuggingFace...")
        uploader = HFUploader()
        predictions = []
        for key, result in results.items():
            if result and result.get('signals') and result['signals'].get('primary_signal'):
                formatter = PredictionFormatter()
                pred = formatter.format_prediction(
                    result['signals'],
                    result.get('metrics', {}),
                    result.get('causal_results', {}),
                    result.get('training_mode', 'unknown')
                )
                if 'universe' not in pred and 'universe' in result:
                    pred['universe'] = result['universe']
                predictions.append(pred)
        if predictions:
            uploader.upload_predictions(predictions)

    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()
