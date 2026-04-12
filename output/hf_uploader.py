"""
HuggingFace Output Module
=========================
Uploads prediction results to HuggingFace datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

try:
    from huggingface_hub import HfApi, create_repo
    HAS_HF = True
except ImportError:
    HAS_HF = False

import config


class HFUploader:
    def __init__(self, repo_name: str = None, token: str = None):
        self.repo_name = repo_name or config.OUTPUT_DATASET
        self.token = token or os.environ.get('HF_TOKEN')
        self.api = None
        if HAS_HF and self.token:
            self.api = HfApi(token=self.token)
        print(f"HFUploader initialized: repo={self.repo_name}, token={'present' if self.token else 'missing'}, HAS_HF={HAS_HF}")

    def prepare_dataset(self, predictions: List[Dict]) -> pd.DataFrame:
        rows = []
        for pred in predictions:
            top_3 = pred.get('top_3_picks', [])
            top_3_tickers = [p.get('ticker', 'N/A') for p in top_3]
            top_3_scores = [p.get('score', 0.0) for p in top_3]
            followers = pred.get('followers', [])
            row = {
                'date': pred.get('date', datetime.now().strftime('%Y-%m-%d')),
                'universe': pred.get('universe', 'unknown'),
                'predicted_leader_etf': pred.get('predicted_leader_etf', 'N/A'),
                'predicted_return': pred.get('predicted_return', 0.0),
                'causal_confidence': pred.get('causal_confidence', 0.0),
                'top_3_picks_tickers': json.dumps(top_3_tickers),
                'top_3_picks_scores': json.dumps(top_3_scores),
                'followers': json.dumps(followers),
                'model_version': pred.get('model_version', config.MODEL_VERSION),
                'training_mode': pred.get('training_mode', 'unknown'),
                'window_start': pred.get('window_start', 'N/A'),
                'window_end': pred.get('window_end', 'N/A'),
                'metrics_annualized_return': pred.get('metrics', {}).get('annualized_return', 0.0),
                'metrics_sharpe_ratio': pred.get('metrics', {}).get('sharpe_ratio', 0.0),
                'metrics_max_drawdown': pred.get('metrics', {}).get('max_drawdown', 0.0),
                'metrics_win_rate': pred.get('metrics', {}).get('win_rate', 0.0),
                'metrics_best_day': pred.get('metrics', {}).get('best_day', 0.0),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"Prepared dataset with {len(df)} rows, columns: {list(df.columns)}")
        return df

    def upload_predictions(self, predictions: List[Dict], commit_message: str = None) -> Dict:
        print("Starting upload_predictions...")
        if not HAS_HF:
            print("huggingface_hub not installed")
            return {'success': False, 'error': 'huggingface_hub not installed'}
        if not self.token:
            print("No HF token found")
            return {'success': False, 'error': 'No HF token'}

        try:
            # Load existing predictions
            existing_df = self.load_existing_predictions()
            print(f"Loaded existing predictions: {len(existing_df)} rows")

            new_df = self.prepare_dataset(predictions)
            print(f"New predictions: {len(new_df)} rows")

            if not existing_df.empty:
                # Convert date columns to string for merging
                existing_df['date'] = existing_df['date'].astype(str)
                new_df['date'] = new_df['date'].astype(str)
                existing_df['_key'] = existing_df['date'] + '_' + existing_df['universe'] + '_' + existing_df['training_mode']
                new_df['_key'] = new_df['date'] + '_' + new_df['universe'] + '_' + new_df['training_mode']
                # Keep only new rows that are not already present
                new_df = new_df[~new_df['_key'].isin(existing_df['_key'])]
                new_df = new_df.drop(columns=['_key'])
                existing_df = existing_df.drop(columns=['_key'])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"Combined dataset: {len(combined_df)} rows (added {len(new_df)} new rows)")
            else:
                combined_df = new_df
                print(f"Using only new dataset: {len(combined_df)} rows")

            if combined_df.empty:
                print("No new predictions to upload")
                return {'success': True, 'repo': self.repo_name, 'n_predictions': 0, 'message': 'No new predictions'}

            # Save locally
            os.makedirs('./output', exist_ok=True)
            local_path = './output/predictions.parquet'
            combined_df.to_parquet(local_path, index=False)
            print(f"Saved parquet locally to {local_path}")

            # Upload to HuggingFace
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo='predictions.parquet',
                repo_id=self.repo_name,
                repo_type='dataset',
                commit_message=commit_message or f'Update predictions: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )
            print("Upload successful")
            return {'success': True, 'repo': self.repo_name, 'n_predictions': len(combined_df), 'message': 'Upload successful'}
        except Exception as e:
            print(f"Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e), 'message': 'Upload failed'}

    def load_existing_predictions(self) -> pd.DataFrame:
        if not HAS_HF or not self.token:
            return pd.DataFrame()
        try:
            from datasets import load_dataset
            ds = load_dataset(self.repo_name, split='train')
            df = ds.to_pandas()
            print(f"Loaded {len(df)} existing predictions from {self.repo_name}")
            return df
        except Exception as e:
            print(f"Could not load existing predictions: {e}")
            return pd.DataFrame()

    def create_repo_if_not_exists(self) -> bool:
        if not HAS_HF or not self.api:
            return False
        try:
            create_repo(self.repo_name, repo_type='dataset', token=self.token, exist_ok=True)
            return True
        except Exception as e:
            print(f"Could not create repo: {e}")
            return False
