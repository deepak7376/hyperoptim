class Hypertune:
    def __init__(self, generations, population_size, model, params):
        self.generations=generations
        self.population_size=population_size
        self.model=model
        self.params=params
        # self.cv=cv
        # self.verbose=verbose
        # self.n_jobs=n_jobs


    def fit(self, X_train, y_train):

        def estimator(individual):
        
            for idx, val in enumerate(individual):
                self.params[list(self.params.keys())[idx]]=val


            self.model.set_params(**self.params)
            self.model.fit(train_X, train_y)

            y_pred = self.model.predict(test_x)
            return mean_squared_error(test_y, y_pred),

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()

        CXPB, MUTPB, POP_SIZE, NGEN = 1, 0.2, self.population_size, self.generations

        h = lambda: self.params

        pop = [creator.Individual([h()[i] for i in list(h().keys())]) for j in range(POP_SIZE)]

        print("\nInitial Population:", pop)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("evaluate", estimator)

        for g in range(NGEN):

            print('\ngeneration:', g)

            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))

            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                print(ind, "---", fit)

            pop[:] = offspring

            best = tools.selBest(pop, POP_SIZE)[0]
            print('best:', best)

        best = tools.selBest(pop, POP_SIZE)[0]
        print('final:',best)
        return best