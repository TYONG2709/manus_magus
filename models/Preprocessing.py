# pre-processing
def pre_process(df):
    df['hand'] = df['hand'].replace('Right', 1)
    df['hand'] = df['hand'].replace('Left', 0)
    df['gesture'] = df['gesture'].replace('thumb_up', 1)
    df['gesture'] = df['gesture'].replace('invalid', 0)

    return df