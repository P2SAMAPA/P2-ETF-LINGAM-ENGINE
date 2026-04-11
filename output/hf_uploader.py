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
            row = {
                'date': pred.get('date'),
                'universe': pred.get('universe'),
                'predicted_leader_etf': pred.get('predicted_leader_etf'),
                'predicted_return': pred.get('predicted_return'),
                'causal_confidence': pred.get('causal_confidence'),
                'top_3_picks_tickers': json.dumps([p['ticker'] for p in pred.get('top_3_picks', [])]),
                'top_3_picks_scores': json.dumps([p.get('score', 0) for p in pred.get('top_3_picks', [])]),
                'followers': json.dumps(pred.get('followers', [])),
                'model_version': pred.get('model_version'),
                'training_mode': pred.get('training_mode'),
                'window_start': pred.get('window_start'),
                'window_end': pred.get('window_end'),
                'metrics_total_return': pred.get('metrics', {}).get('total_return'),
                'metrics_sharpe_ratio': pred.get('metrics', {}).get('sharpe_ratio'),
                'metrics_max_drawdown': pred.get('metrics', {}).get('max_drawdown'),
                'metrics_win_rate': pred.get('metrics', {}).get('win_rate'),
                'metrics_best_day': pred.get('metrics', {}).get('best_day'),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def upload_predictions(
        self,
        predictions: List[Dict],
        commit_message: str = None
    ) -> Dict:
        """
        Upload predictions to HuggingFace.

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
            # Prepare data
            df = self.prepare_dataset(predictions)

            # Save locally first
            local_path = './output/predictions.parquet'
            os.makedirs('./output', exist_ok=True)
            df.to_parquet(local_path)

            # Upload to HF
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo='predictions.parquet',
                repo_id=self.repo_name,
                repo_type='dataset',
                commit_message=commit_message or f'Update predictions: {datetime.now().strftime("%Y-%m-%d")}'
            )

            return {
                'success': True,
                'repo': self.repo_name,
                'n_predictions': len(predictions),
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
            ds = load_dataset(self.repo_name)
            return ds['train'].to_pandas()
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