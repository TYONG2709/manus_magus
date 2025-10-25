# pre-processing
def pre_process(df):
    df['hand'] = df['hand'].replace('Right', 1)
    df['hand'] = df['hand'].replace('Left', 0)

    # 0 0 0 0 - thumb_up
    # 0 0 0 1 - shield
    # 0 0 1 0 - bind
    # 0 1 0 0 - fireball
    # 1 0 0 0 - invalid

    df['gesture'] = df['gesture'].replace('thumb_up', 0)
    df['gesture'] = df['gesture'].replace('shield', 1)
    df['gesture'] = df['gesture'].replace('bind', 10)
    df['gesture'] = df['gesture'].replace('fireball', 100)
    df['gesture'] = df['gesture'].replace('invalid', 1000)

    return df