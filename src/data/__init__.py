"""
Data module
"""
from .preprocessing import ProteinDataProcessor, preprocess_pipeline, split_dataset
from .featurization import (
    OneHotFeatureExtractor,
    ESMFeatureExtractor,
    ProtBERTFeatureExtractor,
    get_feature_extractor,
)
from .dataset import ProteinDataset, ProteinDatasetWithEmbedding, DataLoaderFactory

__all__ = [
    # Preprocessing
    'ProteinDataProcessor',
    'preprocess_pipeline',
    'split_dataset',
    # Featurization
    'OneHotFeatureExtractor',
    'ESMFeatureExtractor',
    'ProtBERTFeatureExtractor',
    'get_feature_extractor',
    # Dataset
    'ProteinDataset',
    'ProteinDatasetWithEmbedding',
    'DataLoaderFactory',
]