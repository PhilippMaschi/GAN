import pandas as pd


def import_and_preprocess_data(path = 'data/all_load_profiles.json'):
    # Import data from JSON file
    df = pd.read_json(path)
    display('Initial dataframe', df)

    # Set 'Date' column as index
    df.set_index('Date', inplace = True)

    # Convert remaining columns to float64
    df = df.astype('float64')

    # Identify zero-consumption profiles
    zeroConsumProfiles = df.columns[df.sum() == 0].values
    display('Zero-consumption profiles', zeroConsumProfiles)

    # Remove zero-consumption profiles
    df.drop(columns = zeroConsumProfiles, inplace = True)

    # Identify duplicated timestamps
    dupeTS_df = df[df.index.duplicated(keep = False)]
    display('Duplicated timestamps', dupeTS_df)

    # Remove duplicated timestamps (keep first occurance)
    df = df[~df.index.duplicated()]

    # Reindex dataframe (sorts index and adds missing timestamps)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq = 'H'))

    # Identify missing timestamps
    missTS = df[df.isna().any(axis = 1)]
    missTS.index = missTS.index.strftime('%Y-%m-%d %H:%M:%S')
    display('Missing timestamps', missTS)

    # Fill missing timestamps (propagate last valid observation forward)
    df.fillna(method = 'ffill', inplace = True)

    # Rename index
    df.index.name = 'timestamp'

    # Reset index
    df.reset_index(inplace = True)

    # Return preprocessed dataframe
    return df