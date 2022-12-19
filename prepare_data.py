def create_timestamp(df, date_column) -> pd.DataFrame:
    """creates a timestamp in the 'date' column so we can cluster the data more easily"""
    df.loc[:, date_column] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
    return df


# function to convert categorical variables in numerical. Necessary ???
def from_cat_to_num(df) -> pd.DataFrame:
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    return df

def drop_missing_values(col_with_missing_val):
    data_no_missing_val = data.drop(columns=col_with_missing_val, axis=1)
    data_no_missing_val.to_csv('data_clean.csv', index=False)
    return data_no_missing_val