if __name__=="__main__":
    from sklearn.ensemble import GradientBoostingRegressor


    train_X = np.random.randint(2, 30, (10,2))
    train_y = np.random.randint(2, 30, (10))
    test_x = np.random.randint(2, 30, (5,2))
    test_y = np.random.randint(2, 30, (5))

    clf = AdaBoostRegressor()

    params = {
    'learning_rate': np.random.choice([0.1, .2, 0.3, 0.5, 0.6]),
    'n_estimators': np.random.choice([x for x in range(100, 600, 10)])
}


    model = Hypertune(generations=10, population_size=10, model=clf, params=params)
    model.fit(train_X, train_y)


