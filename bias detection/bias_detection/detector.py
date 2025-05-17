import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from typing import Dict, List, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BiasDetector:
    """Main class for detecting various types of bias in training data."""
    
    def __init__(self):
        self.metrics = {}
        self.tokenizer = None
        self.model = None
    
    def analyze_dataset(self, 
                       data: pd.DataFrame,
                       protected_attributes: List[str],
                       target_column: str,
                       text_columns: Optional[List[str]] = None) -> Dict:
        """
        Analyze dataset for various types of bias.
        
        Args:
            data: Input DataFrame
            protected_attributes: List of columns containing protected attributes
            target_column: Name of the target variable column
            text_columns: Optional list of columns containing text data
            
        Returns:
            Dictionary containing bias metrics
        """
        self.metrics = {}
        
        # Statistical bias detection
        self.metrics['statistical'] = self._detect_statistical_bias(
            data, protected_attributes, target_column
        )
        
        # Representation bias detection
        self.metrics['representation'] = self._detect_representation_bias(
            data, protected_attributes
        )
        
        # If text columns are provided, perform text bias detection
        if text_columns:
            self.metrics['text'] = self._detect_text_bias(data[text_columns])
        
        return self.metrics
    
    def _detect_statistical_bias(self,
                               data: pd.DataFrame,
                               protected_attributes: List[str],
                               target_column: str) -> Dict:
        """Detect statistical bias using metrics like disparate impact and statistical parity."""
        stats = {}
        
        for attr in protected_attributes:
            # Define privileged and unprivileged groups
            value_counts = data[attr].value_counts()
            privileged_value = value_counts.index[0]
            unprivileged_values = value_counts.index[1:]
            
            # Convert to AIF360 dataset format
            dataset = BinaryLabelDataset(
                df=data,
                label_names=[target_column],
                protected_attribute_names=[attr],
                privileged_protected_attributes=[[privileged_value]],
                unprivileged_protected_attributes=[unprivileged_values.tolist()]
            )
            
            metrics = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=[{attr: val} for val in unprivileged_values],
                privileged_groups=[{attr: privileged_value}]
            )
            
            # Calculate available metrics
            stats[attr] = {
                'disparate_impact': metrics.disparate_impact(),
                'statistical_parity_difference': metrics.statistical_parity_difference(),
                'privileged_group': privileged_value,
                'unprivileged_groups': unprivileged_values.tolist(),
                'num_positives_privileged': metrics.num_positives(privileged=True),
                'num_positives_unprivileged': metrics.num_positives(privileged=False),
                'num_negatives_privileged': metrics.num_negatives(privileged=True),
                'num_negatives_unprivileged': metrics.num_negatives(privileged=False),
                'base_rate_privileged': metrics.base_rate(privileged=True),
                'base_rate_unprivileged': metrics.base_rate(privileged=False)
            }
        
        return stats
    
    def _detect_representation_bias(self,
                                  data: pd.DataFrame,
                                  protected_attributes: List[str]) -> Dict:
        """Detect representation bias by analyzing distribution of protected attributes."""
        representation = {}
        
        for attr in protected_attributes:
            # Calculate distribution of protected attribute values
            distribution = data[attr].value_counts(normalize=True)
            
            # Calculate entropy as a measure of diversity
            entropy = -np.sum(distribution * np.log(distribution))
            
            representation[attr] = {
                'distribution': distribution.to_dict(),
                'entropy': entropy,
                'min_representation': distribution.min(),
                'max_representation': distribution.max()
            }
        
        return representation
    
    def _detect_text_bias(self, text_data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """Detect bias in text data using pre-trained language models."""
        if self.tokenizer is None:
            # Load pre-trained model for bias detection
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base")
        
        text_bias = {
            'toxic_content': [],
            'sentiment_bias': [],
            'gender_bias': []
        }
        
        # Process text in batches
        batch_size = 32
        for i in range(0, len(text_data), batch_size):
            batch = text_data.iloc[i:i+batch_size]
            
            # Tokenize and get model predictions
            inputs = self.tokenizer(batch.tolist(), 
                                  padding=True, 
                                  truncation=True, 
                                  return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
            
            # Analyze predictions for different types of bias
            text_bias['toxic_content'].extend(predictions[:, 1].tolist())
        
        return {
            'mean_toxic_score': np.mean(text_bias['toxic_content']),
            'max_toxic_score': np.max(text_bias['toxic_content']),
            'num_potentially_biased': sum(score > 0.5 for score in text_bias['toxic_content'])
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable report of detected biases."""
        if not self.metrics:
            return "No bias analysis has been performed yet."
        
        report = []
        report.append("Bias Detection Report")
        report.append("===================")
        
        # Statistical bias summary
        if 'statistical' in self.metrics:
            report.append("\nStatistical Bias:")
            for attr, metrics in self.metrics['statistical'].items():
                report.append(f"\nProtected Attribute: {attr}")
                report.append(f"- Privileged Group: {metrics['privileged_group']}")
                report.append(f"- Unprivileged Groups: {', '.join(map(str, metrics['unprivileged_groups']))}")
                report.append(f"- Disparate Impact: {metrics['disparate_impact']:.3f}")
                report.append(f"- Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f}")
                report.append("\nDetailed Metrics:")
                report.append(f"- Privileged Group Positive Rate: {metrics['base_rate_privileged']:.3f}")
                report.append(f"- Unprivileged Group Positive Rate: {metrics['base_rate_unprivileged']:.3f}")
                report.append(f"- Privileged Group Positives: {metrics['num_positives_privileged']}")
                report.append(f"- Unprivileged Group Positives: {metrics['num_positives_unprivileged']}")
                report.append(f"- Privileged Group Negatives: {metrics['num_negatives_privileged']}")
                report.append(f"- Unprivileged Group Negatives: {metrics['num_negatives_unprivileged']}")
        
        # Representation bias summary
        if 'representation' in self.metrics:
            report.append("\nRepresentation Bias:")
            for attr, metrics in self.metrics['representation'].items():
                report.append(f"\nAttribute: {attr}")
                report.append(f"- Entropy: {metrics['entropy']:.3f}")
                report.append(f"- Min Representation: {metrics['min_representation']:.3f}")
                report.append(f"- Max Representation: {metrics['max_representation']:.3f}")
                report.append("\nDistribution:")
                for val, prop in metrics['distribution'].items():
                    report.append(f"  {val}: {prop:.3f}")
        
        # Text bias summary
        if 'text' in self.metrics:
            report.append("\nText Bias:")
            report.append(f"- Mean Toxic Score: {self.metrics['text']['mean_toxic_score']:.3f}")
            report.append(f"- Max Toxic Score: {self.metrics['text']['max_toxic_score']:.3f}")
            report.append(f"- Number of Potentially Biased Texts: {self.metrics['text']['num_potentially_biased']}")
        
        return "\n".join(report) 