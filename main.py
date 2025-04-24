# from src.preprocessing.clean_data import load_and_validate_data, save_clean_data
# from src.features.extract_metrics import extract_features
# import pandas as pd

# def main():
#     # Step 1: Load raw -> Clean it
#     df = load_and_validate_data('data/raw/t1.csv')
#     if df is not None:
#         save_clean_data(df)

#         # Step 2: Extract features from cleaned data
#         df_features = extract_features(df)
#         print("✅ Feature extraction successful!")
#         print(df_features.head())  # preview

#         # Step 3: Save to processed/featured_data.csv
#         df_features.to_csv('data/processed/featured_data.csv', index=False)
#         print("📁 Saved featured data to data/processed/featured_data.csv")

# if __name__ == "__main__":
#     main()
