def split_data(data, validation_cutoff, test_cutoff):
    # Training set
    X_train = data[data['datetime'] < validation_cutoff].drop('target', axis=1).set_index('datetime')
    y_train = data[data['datetime'] < validation_cutoff][['datetime', 'target']].set_index('datetime')

    # Validation set
    X_valid = data[(data['datetime'] >= validation_cutoff) & (data['datetime'] < test_cutoff)].drop('target', axis=1).set_index('datetime')
    y_valid = data[(data['datetime'] >= validation_cutoff) & (data['datetime'] < test_cutoff)][['datetime', 'target']].set_index('datetime')

    # Test set
    X_test = data[data['datetime'] >= test_cutoff].drop('target', axis=1).set_index('datetime')
    y_test = data[data['datetime'] >= test_cutoff][['datetime', 'target']].set_index('datetime')

    return X_train, y_train, X_valid, y_valid, X_test, y_test