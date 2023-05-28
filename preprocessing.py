def remove_incomplete_days(df):
    temp_df = df.groupby('date').count()
    incompleteDays = temp_df[(temp_df < 24).any(axis = 1)].index
    df = df.loc[~df['date'].isin(incompleteDays)]
    return df