import pandas as pd
import numpy as np
from bias_detection.detector import BiasDetector
from bias_mitigation.mitigator import BiasMitigator
from sklearn.preprocessing import LabelEncoder

def create_sample_dataset(size=1000):
    """Create a sample dataset with potential biases for demonstration."""
    np.random.seed(42)
    
    # Create biased dataset
    data = {
        'age': np.random.normal(35, 10, size),
        'gender': np.random.choice(['M', 'F'], size, p=[0.7, 0.3]),  # Gender bias
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], size, p=[0.6, 0.2, 0.1, 0.1]),  # Racial bias
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], size),
        'experience': np.random.normal(10, 5, size),
        'salary': np.zeros(size)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce biased salary calculations
    for idx in df.index:
        base_salary = 50000
        
        # Add gender bias
        if df.loc[idx, 'gender'] == 'M':
            base_salary *= 1.2
        
        # Add racial bias
        if df.loc[idx, 'race'] == 'White':
            base_salary *= 1.15
        
        # Add education factor
        education_factor = {
            'HS': 1.0,
            'Bachelor': 1.3,
            'Master': 1.5,
            'PhD': 1.8
        }
        base_salary *= education_factor[df.loc[idx, 'education']]
        
        # Add experience factor
        base_salary *= (1 + df.loc[idx, 'experience'] * 0.03)
        
        df.loc[idx, 'salary'] = base_salary
    
    # Convert salary to binary classification (high/low)
    median_salary = df['salary'].median()
    df['high_salary'] = (df['salary'] > median_salary).astype(int)
    
    return df

def encode_categorical_variables(df, categorical_columns):
    """Encode categorical variables and store the encoders for reference."""
    df_encoded = df.copy()
    encoders = {}
    
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df_encoded[column] = encoders[column].fit_transform(df[column])
    
    return df_encoded, encoders

def main():
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Encode categorical variables
    categorical_columns = ['gender', 'race', 'education']
    df_encoded, encoders = encode_categorical_variables(df, categorical_columns)
    
    # Print encoding mappings
    print("\nEncoding mappings:")
    for column, encoder in encoders.items():
        print(f"\n{column} encoding:")
        for i, label in enumerate(encoder.classes_):
            print(f"{label} -> {i}")
    
    # Initialize bias detector
    print("\nInitializing bias detection...")
    detector = BiasDetector()
    
    # Analyze dataset for bias
    protected_attributes = ['gender', 'race']
    target_column = 'high_salary'
    
    print("Analyzing dataset for bias...")
    bias_report = detector.analyze_dataset(
        data=df_encoded,
        protected_attributes=protected_attributes,
        target_column=target_column
    )
    
    # Print bias detection report
    print("\nBias Detection Report:")
    print(detector.generate_report())
    
    # Initialize bias mitigator
    print("\nInitializing bias mitigation...")
    mitigator = BiasMitigator()
    
    # Apply bias mitigation
    print("Applying bias mitigation strategies...")
    mitigated_df = mitigator.mitigate(
        data=df_encoded,
        bias_report=bias_report,
        protected_attributes=protected_attributes,
        target_column=target_column
    )
    
    # Print mitigation report
    print("\nBias Mitigation Report:")
    print(mitigator.generate_report())
    
    # Compare original and mitigated datasets
    print("\nDataset Comparison:")
    print("\nOriginal Dataset Statistics:")
    for attr in protected_attributes:
        print(f"\n{attr} distribution:")
        print(df[attr].value_counts(normalize=True))
        print(f"\n{attr} vs {target_column} correlation:")
        print(df_encoded[[attr, target_column]].corr().iloc[0, 1])
    
    print("\nMitigated Dataset Statistics:")
    for attr in protected_attributes:
        print(f"\n{attr} distribution:")
        # Convert encoded values back to original categories
        mitigated_attr = pd.Series(
            encoders[attr].inverse_transform(mitigated_df[attr].astype(int)),
            index=mitigated_df.index
        )
        print(mitigated_attr.value_counts(normalize=True))
        print(f"\n{attr} vs {target_column} correlation:")
        print(mitigated_df[[attr, target_column]].corr().iloc[0, 1])

if __name__ == "__main__":
    main() 