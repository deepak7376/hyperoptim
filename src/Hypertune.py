import random
from deap import creator, base, tools, algorithms


class GeneticAlgorithmSearch:

    def __init__(self, model_builder, objective, max_epochs, directory, project_name):
        self.model_builder = model_builder
        self.objective = objective
        self.max_epochs = max_epochs
        self.directory = directory
        self.project_name = project_name
        self.best_hp = None

    def search(self, x_train, y_label, params, epochs, validation_split, callbacks):
        self._search_hyperparameter(x_train, y_label, params, epochs, validation_split)

    def get_best_hyperparameters(self, num_trials=1):
        return self.best_hp

    def build(self, best_hps):
        return self.model_builder(best_hps)
    
    def _search_hyperparameter(self, x_train, y_label, params, epochs, validation_split):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def _eval_val_accuracy(offspring):
            model = self.model_builder(offspring)
            history = model.fit(x_train, y_label, epochs=epochs, validation_split=validation_split, verbose=2)
            val_acc_per_epoch = history.history['val_accuracy']
            print(f"max val accuracy: {max(val_acc_per_epoch)}")
            return max(val_acc_per_epoch),

        def _individual():
            return creator.Individual(params)

        toolbox = base.Toolbox()
        toolbox.register("individual", _individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        toolbox.register("evaluate", _eval_val_accuracy)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=5)

        NGEN=5
        for gen in range(NGEN):
            print(f"GENERATION {gen} STARTED")
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                print(f"HP: {ind}, SCORE: {fit}")
                ind.fitness.values = fit
            print(f"GENERATION {gen} COMPLETED, BEST_SCORE: {tools.selBest(population, k=1)}, BEST_HP: {tools.selBest(population, k=1)}")
            population = toolbox.select(offspring, k=len(population))
        
        self.best_hp = tools.selBest(population, k=1)


class Hparams:
    @staticmethod
    def Int(param_name, min_value, max_value, step):
        return random.randrange(min_value, max_value, step)
    
    @staticmethod
    def Choice(param_name, values):
        return random.choice(values)


if __name__ == "__main__":
    ht = Hparams()
    # define search parameters
    hp_units = ht.Int('units', min_value=32, max_value=512, step=32)
    hp_learning_rate = ht.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    params = [hp_units, hp_learning_rate]
    print(params)