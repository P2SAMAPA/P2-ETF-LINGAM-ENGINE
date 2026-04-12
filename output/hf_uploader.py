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
    """
    Manages upload of prediction results to HuggingFace.
    """

    def __init__(
        self,
        repo_name: str = None,
        token: str = None
    ):
        """
        Initialize the HF uploader.

        Args:
            repo_name: HuggingFace repository name
            token: HF token for authentication
        """
        self.repo_name = repo_name or config.OUTPUT_DATASET
        self.token = token or os.environ.get('HF_TOKEN')
        self.api = None

        if HAS_HF and self.token:
            self.api = HfApi(token=self.token)

    def prepare_dataset(
        self,
        predictions: List[Dict],
        split: str = 'train'
    ) -> pd.DataFrame:
        """
        Prepare predictions as a HuggingFace dataset.

        Args:
            predictions: List of prediction dictionaries
            split: Dataset split name

        Returns:
            DataFrame ready for upload
        """
        rows = []

        for pred in predictions:
            # Safely extract top_3_picks
            top_3 = pred.get('top_3_picks', [])
            if not isinstance(top_3, list):
                top_3 = []
            top_3_tickers = [p.get('ticker', 'N/A') for p in top_3]
            top_3_scores = [p.get('score', 0.0) for p in top_3]

            # Safely extract followers
            followers = pred.get('followers', [])
            if not isinstance(followers, list):
                followers = []

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
                'metrics_total_return': pred.get('metrics', {}).get('total_return', 0.0),
                'metrics_sharpe_ratio': pred.get('metrics', {}).get('sharpe_ratio', 0.0),
                'metrics_max_drawdown': pred.get('metrics', {}).get('max_drawdown', 0.0),
                'metrics_win_rate': pred.get('metrics', {}).get('win_rate', 0.0),
                'metrics_best_day': pred.get('metrics', {}).get('best_day', 0.0),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def upload_predictions(
        self,
        predictions: List[Dict],
        commit_message: str = None
    ) -> Dict:
        """
        Upload predictions to HuggingFace, appending to existing data.

        Args:
            predictions: List of prediction dictionaries
            commit_message: Optional commit message

        Returns:
            Dictionary with upload status
        """
        if not HAS_HF:
            return {
                'success': False,
                'error': 'huggingface_hub not installed',
                'message': 'Install with: pip install huggingface_hub'
            }

        if not self.token:
            return {
                'success': False,
                'error': 'No HF token',
                'message': 'Set HF_TOKEN environment variable'
            }

        try:
            # Load existing predictions if any
            existing_df = self.load_existing_predictions()
            new_df = self.prepare_dataset(predictions)

            # Combine: remove duplicates based on (date, universe, training_mode) if needed
            if not existing_df.empty:
                # Ensure date column is string for comparison
                existing_df['date'] = existing_df['date'].astype(str)
                new_df['date'] = new_df['date'].astype(str)
                # Create a key
                existing_df['_key'] = existing_df['date'] + '_' + existing_df['universe'] + '_' + existing_df['training_mode']
                new_df['_key'] = new_df['date'] + '_' + new_df['universe'] + '_' + new_df['training_mode']
                # Keep only new rows that are not already present
                new_df = new_df[~new_df['_key'].isin(existing_df['_key'])]
                # Drop temporary key columns
                new_df = new_df.drop(columns=['_key'])
                existing_df = existing_df.drop(columns=['_key'])
                # Concatenate
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            if combined_df.empty:
                return {
                    'success': True,
                    'repo': self.repo_name,
                    'n_predictions': 0,
                    'message': 'No new predictions to upload'
                }

            # Save locally
            local_path = './output/predictions.parquet'
            os.makedirs('./output', exist_ok=True)
            combined_df.to_parquet(local_path, index=False)

            # Upload to HF
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo='predictions.parquet',
                repo_id=self.repo_name,
                repo_type='dataset',
                commit_message=commit_message or f'Update predictions: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )

            return {
                'success': True,
                'repo': self.repo_name,
                'n_predictions': len(combined_df),
                'message': 'Upload successful'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Upload failed'
            }

    def load_existing_predictions(self) -> pd.DataFrame:
        """
        Load existing predictions from HuggingFace.

        Returns:
            DataFrame of existing predictions
        """
        if not HAS_HF or not self.token:
            return pd.DataFrame()

        try:
            from datasets import load_dataset
            ds = load_dataset(self.repo_name, split='train')
            df = ds.to_pandas()
            return df
        except Exception as e:
            print(f"Could not load existing predictions: {e}")
            return pd.DataFrame()

    def create_repo_if_not_exists(self) -> bool:
        """
        Create the repository if it doesn't exist.

        Returns:
            True if successful, False otherwise
        """
        if not HAS_HF or not self.api:
            return False

        try:
            create_repo(
                self.repo_name,
                repo_type='dataset',
                token=self.token,
                exist_ok=True
            )
            return True
        except Exception as e:
            print(f"Could not create repo: {e}")
            return False
