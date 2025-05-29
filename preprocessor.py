"""
Data preprocessing module for the trading bot
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

class DataPreprocessor:
    """Data preprocessing class for ML pipeline"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize data preprocessor
        
        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config or {
            'scale_features': True,
            'feature_selection': True,
            'n_features': 20,
            'impute_strategy': 'mean',
            'outlier_threshold': 3.0
        }
        self._setup_logging()
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        self.selected_features = None
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def preprocess_data(self, data: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data for ML training
        
        Args:
            data: DataFrame containing features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (X_processed, y) where X_processed is the processed features
            and y is the target variable
        """
        try:
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Remove outliers
            X = self._remove_outliers(X)
            
            # Scale features
            if self.config['scale_features']:
                X = self._scale_features(X)
                
            # Select features
            if self.config['feature_selection']:
                X = self._select_features(X, y)
                
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise
            
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Create imputer if not already created
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy=self.config['impute_strategy'])
            
        # Fit and transform
        imputed_data = self.imputer.fit_transform(data)
        return pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
        
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[(z_scores < self.config['outlier_threshold']).all(axis=1)]
        
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        # Create scaler if not already created
        if self.scaler is None:
            self.scaler = StandardScaler()
            
        # Fit and transform
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        
    def _select_features(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features using SelectKBest"""
        # Create selector if not already created
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.config['n_features'], data.shape[1])
            )
            
        # Fit and transform
        selected_data = self.feature_selector.fit_transform(data, target)
        
        # Store selected feature names
        self.selected_features = data.columns[self.feature_selector.get_support()]
        
        return pd.DataFrame(selected_data, columns=self.selected_features, index=data.index)
        
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
        """
        if self.imputer is None or self.scaler is None:
            raise ValueError("Preprocessors must be fitted first")
            
        # Handle missing values
        data = pd.DataFrame(
            self.imputer.transform(data),
            columns=data.columns,
            index=data.index
        )
        
        # Scale features
        if self.config['scale_features']:
            data = pd.DataFrame(
                self.scaler.transform(data),
                columns=data.columns,
                index=data.index
            )
            
        # Select features if feature selection was performed
        if self.config['feature_selection'] and self.selected_features is not None:
            data = data[self.selected_features]
            
        return data
        
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if self.feature_selector is None:
            raise ValueError("Feature selector must be fitted first")
            
        scores = self.feature_selector.scores_
        return pd.Series(scores, index=self.selected_features).sort_values(ascending=False)
        
    def get_preprocessing_pipeline(self) -> Pipeline:
        """Get the complete preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['impute_strategy'])),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.selected_features)
            ]
        )
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selector', SelectKBest(
                score_func=f_classif,
                k=self.config['n_features']
            ))
        ]) 