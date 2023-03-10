metSeason_dict = \
    {item: 'Spring' for item in [3, 4, 5]} | \
    {item: 'Summer' for item in [6, 7, 8]} | \
    {item: 'Fall' for item in [9, 10, 11]} | \
    {item: 'Winter' for item in [12, 1, 2]}

weekend_dict = \
    {item: False for item in range(5)} | \
    {item: True for item in [5, 6]}