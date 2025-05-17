import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.utils import resample
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

class BiasMitigator:
    """Class for implementing various bias mitigation strategies."""
    
    def __init__(self):
        self.original_data = None
        self.mitigation_stats = {}
    
    def mitigate(self,
                 data: pd.DataFrame,
                 bias_report: Dict,
                 protected_attributes: List[str],
                 target_column: str,
                 strategies: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply bias mitigation strategies to the dataset.
        
        Args:
            data: Input DataFrame
            bias_report: Dictionary containing bias metrics from BiasDetector
            protected_attributes: List of protected attribute columns
            target_column: Name of the target variable column
            strategies: List of mitigation strategies to apply
            
        Returns:
            DataFrame with mitigated bias
        """
        self.original_data = data.copy()
        mitigated_data = data.copy()
        
        if strategies is None:
            strategies = ['reweighting', 'resampling', 'transformation']
        
        for strategy in strategies:
            if strategy == 'reweighting':
                mitigated_data = self._apply_reweighting(
                    mitigated_data,
                    protected_attributes,
                    target_column
                )
            elif strategy == 'resampling':
                mitigated_data = self._apply_resampling(
                    mitigated_data,
                    protected_attributes,
                    target_column
                )
            elif strategy == 'transformation':
                mitigated_data = self._apply_transformation(
                    mitigated_data,
                    protected_attributes,
                    target_column
                )
        
        return mitigated_data
    
    def _apply_reweighting(self,
                          data: pd.DataFrame,
                          protected_attributes: List[str],
                          target_column: str) -> pd.DataFrame:
        """Apply reweighting to balance protected attributes."""
        reweighted_data = data.copy()
        
        for attr in protected_attributes:
            # Convert to AIF360 dataset format
            dataset = BinaryLabelDataset(
                df=data,
                label_names=[target_column],
                protected_attribute_names=[attr]
            )
            
            # Apply reweighting
            reweighing = Reweighing(unprivileged_groups=[{attr: 0}],
                                   privileged_groups=[{attr: 1}])
            transformed_dataset = reweighing.fit_transform(dataset)
            
            # Extract sample weights
            reweighted_data[f'{attr}_weight'] = transformed_dataset.instance_weights
        
        self.mitigation_stats['reweighting'] = {
            'attributes': protected_attributes,
            'weight_ranges': {
                attr: {
                    'min': reweighted_data[f'{attr}_weight'].min(),
                    'max': reweighted_data[f'{attr}_weight'].max(),
                    'mean': reweighted_data[f'{attr}_weight'].mean()
                } for attr in protected_attributes
            }
        }
        
        return reweighted_data
    
    def _apply_resampling(self,
                         data: pd.DataFrame,
                         protected_attributes: List[str],
                         target_column: str) -> pd.DataFrame:
        """Apply resampling to balance protected attributes."""
        resampled_dfs = []
        
        for attr in protected_attributes:
            # Get value counts for the protected attribute
            value_counts = data[attr].value_counts()
            majority_class = value_counts.index[0]
            majority_size = value_counts.max()
            
            # Resample each minority class
            for val in value_counts.index[1:]:
                minority_df = data[data[attr] == val]
                if len(minority_df) < majority_size:
                    resampled_minority = resample(
                        minority_df,
                        replace=True,
                        n_samples=majority_size,
                        random_state=42
                    )
                    resampled_dfs.append(resampled_minority)
            
            # Add majority class
            resampled_dfs.append(data[data[attr] == majority_class])
        
        # Combine all resampled data
        resampled_data = pd.concat(resampled_dfs, axis=0)
        
        self.mitigation_stats['resampling'] = {
            'original_size': len(data),
            'resampled_size': len(resampled_data),
            'attribute_distributions': {
                attr: resampled_data[attr].value_counts().to_dict()
                for attr in protected_attributes
            }
        }
        
        return resampled_data
    
    def _apply_transformation(self,
                            data: pd.DataFrame,
                            protected_attributes: List[str],
                            target_column: str) -> pd.DataFrame:
        """Apply feature transformation to reduce bias."""
        transformed_data = data.copy()
        
        for attr in protected_attributes:
            # Calculate correlation between protected attribute and other features
            correlations = data.corr()[attr].abs().sort_values(ascending=False)
            
            # Identify highly correlated features (excluding protected attributes and target)
            high_corr_features = correlations[
                (correlations > 0.5) &
                (~correlations.index.isin(protected_attributes + [target_column]))
            ].index
            
            # Apply transformation to highly correlated features
            for feature in high_corr_features:
                if data[feature].dtype in ['int64', 'float64']:
                    # Standardize the feature within each protected attribute group
                    for val in data[attr].unique():
                        mask = transformed_data[attr] == val
                        group_data = transformed_data.loc[mask, feature]
                        transformed_data.loc[mask, feature] = (
                            (group_data - group_data.mean()) / group_data.std()
                        )
        
        self.mitigation_stats['transformation'] = {
            'transformed_features': list(high_corr_features),
            'correlation_reduction': {
                feature: {
                    'before': correlations[feature],
                    'after': transformed_data.corr()[attr].abs()[feature]
                } for feature in high_corr_features
            }
        }
        
        return transformed_data
    
    def get_mitigation_stats(self) -> Dict:
        """Return statistics about the applied mitigation strategies."""
        return self.mitigation_stats
    
    def generate_report(self) -> str:
        """Generate a human-readable report of the mitigation results."""
        if not self.mitigation_stats:
            return "No bias mitigation has been performed yet."
        
        report = []
        report.append("Bias Mitigation Report")
        report.append("=====================")
        
        if 'reweighting' in self.mitigation_stats:
            report.append("\nReweighting Results:")
            for attr in self.mitigation_stats['reweighting']['attributes']:
                weights = self.mitigation_stats['reweighting']['weight_ranges'][attr]
                report.append(f"\nAttribute: {attr}")
                report.append(f"- Min weight: {weights['min']:.3f}")
                report.append(f"- Max weight: {weights['max']:.3f}")
                report.append(f"- Mean weight: {weights['mean']:.3f}")
        
        if 'resampling' in self.mitigation_stats:
            report.append("\nResampling Results:")
            report.append(f"- Original dataset size: {self.mitigation_stats['resampling']['original_size']}")
            report.append(f"- Resampled dataset size: {self.mitigation_stats['resampling']['resampled_size']}")
            
            for attr, dist in self.mitigation_stats['resampling']['attribute_distributions'].items():
                report.append(f"\nAttribute: {attr} distribution:")
                for val, count in dist.items():
                    report.append(f"- {val}: {count}")
        
        if 'transformation' in self.mitigation_stats:
            report.append("\nTransformation Results:")
            report.append("Transformed features:")
            for feature in self.mitigation_stats['transformation']['transformed_features']:
                corr = self.mitigation_stats['transformation']['correlation_reduction'][feature]
                report.append(f"\n{feature}:")
                report.append(f"- Correlation before: {corr['before']:.3f}")
                report.append(f"- Correlation after: {corr['after']:.3f}")
        
        return "\n".join(report) 