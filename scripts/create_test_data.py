"""
Script to create test dataset from your preprocessed data
"""
import pandas as pd
import joblib
from pathlib import Path


def create_test_dataset_from_artifacts(
    artifacts_dir: str = "artifacts",
    output_path: str = "tests/real_test_data.csv",
    num_samples: int = 500
):
    """
    Create test dataset from the preprocessed CIC-IDS2017 data
    that was used for training
    """
    import json
    
    artifacts_dir = Path(artifacts_dir)
    
    # Load transform meta to get column names
    with open(artifacts_dir / "transform_meta.json", 'r') as f:
        meta = json.load(f)
    
    cols = meta['cols']
    print(f"Required columns: {len(cols)}")
    
    # Try to load the preprocessed data if available
    data_path = artifacts_dir / "lite_clean_data_collapsed.pkl"
    
    if data_path.exists():
        print(f"Loading data from: {data_path}")
        data = joblib.load(data_path)
        
        if isinstance(data, dict):
            X = data.get('X')
            y = data.get('y')
            
            if X is not None:
                # Create DataFrame
                if hasattr(X, 'columns'):
                    df = X
                else:
                    df = pd.DataFrame(X, columns=cols)
                
                # Sample
                if len(df) > num_samples:
                    sample_idx = df.sample(n=num_samples, random_state=42).index
                    df = df.loc[sample_idx]
                    if y is not None:
                        y = y[sample_idx] if hasattr(y, '__getitem__') else y. loc[sample_idx]
                
                # Add label if available
                if y is not None:
                    # Load label map
                    with open(artifacts_dir / "label_map.json", 'r') as f:
                        label_data = json.load(f)
                    id_to_label = {int(k): v for k, v in label_data['id_to_label'].items()}
                    
                    df['actual_label'] = [id_to_label.get(int(yi), 'Unknown') for yi in y]
                
                # Save
                df.to_csv(output_path, index=False)
                print(f"✅ Test dataset saved to: {output_path}")
                print(f"   Samples: {len(df)}")
                print(f"   Columns: {len(df.columns)}")
                
                if 'actual_label' in df. columns:
                    print(f"   Label distribution:")
                    print(df['actual_label'].value_counts())
                
                return df
    
    print("❌ Could not find preprocessed data.  Please provide path to CIC-IDS2017 data.")
    return None


if __name__ == "__main__":
    create_test_dataset_from_artifacts()