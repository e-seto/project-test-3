import pandas as pd

def preprocess_csv(input_path, output_path):
    """
    Clean a CSV transaction file:
      1. Read with pipe (|) delimiter.
      2. Drop personally identifiable information (PII).
      3. Parse date and time columns and combine into a single datetime.
      4. Convert date of birth to datetime and compute age at transaction.
      5. Drop redundant columns and sort by transaction datetime.
    """

    # Read the raw data (pipe delimited)
    df = pd.read_csv(input_path, sep='|')

    # Drop PII columns
    pii_cols = [
        'ssn', 'first', 'last', 'street', 'acct_num', 'profile'
    ]
    df = df.drop(columns=pii_cols)

    # Remove fraud_ prefix from merchant names 
    df['merchant'] = df['merchant'].str.replace(r'^fraud_', '', regex=True)

    # Convert transaction date
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')

    # Parse transaction time
    trans_time_parsed = pd.to_datetime(
        df['trans_time'], format='%H:%M:%S', errors='coerce'
    )

    # Combine date + time
    df['trans_datetime'] = pd.to_datetime(
        df['trans_date'].dt.date.astype(str) + ' ' + trans_time_parsed.dt.time.astype(str),
        errors='coerce'
    )

    # Convert DOB and compute age
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = (df['trans_datetime'] - df['dob']).dt.days // 365

    # Drop redundant columns now
    df = df.drop(columns=['trans_time', 'unix_time', 'trans_date', 'dob', 'trans_num'])

    # Optimize category column
    df['category'] = df['category'].astype('category')

    # Sort by time
    df = df.sort_values('trans_datetime').reset_index(drop=True)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print("Cleaned file saved to:", output_path)
    print("Rows:", df.shape[0], "| Columns:", df.shape[1])

    return df

if __name__ == "__main__":
    input_file = "path/to/raw_transactions.csv"
    output_file = "data/raw/cleaned_output.csv"

    preprocess_csv(input_file, output_file)