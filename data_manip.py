def remove_incomplete_days(df):
    df_temp = df.groupby('date').count()
    incompleteDays = df_temp[(df_temp < 24).any(axis = 1)].index
    df = df.loc[~df['date'].isin(incompleteDays)]
    incompleteDays_list = [item.strftime('%Y-%m-%d')for item in incompleteDays.tolist()]
    print(f'The following days were removed: {incompleteDays_list}')
    return df