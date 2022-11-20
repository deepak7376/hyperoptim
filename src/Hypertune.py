import random
from deap import creator, base, tools, algorithms


class GASearch:
    def __init__(
        self,
        model_builder,
        params,
        objective,
        weights=(1.0,),
        pop_size=5,
        gen=5,
        max_epochs=10,
        directory="\home",
        project_name="search_hyperparam",
    ):
        self.model_builder = model_builder
        self.params = params
        self.objective = objective  # ('loss', 'accuracy', 'val_loss', 'val_accuracy')
        self.weights = weights  # (1.0, -1.0)
        self.pop_size = pop_size
        self.gen = gen
        self.max_epochs = max_epochs
        self.directory = directory
        self.project_name = project_name
        self.best_pop = None

    def search(
        self,
        x_train=None,
        y_train=None,
        batch_size=None,
        epochs=2,
        verbose="auto",
        callbacks=None,
        validation_split=0.2,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        creator.create("Fitness", base.Fitness, weights=self.weights)
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox = base.Toolbox()

        individual = lambda: creator.Individual([param() for param in self.params])

        def objective_fn(params):
            model = self.model_builder(params)
            hist = model.fit(
                x_train, y_train, epochs=epochs, validation_split=validation_split
            )
            best_score = max(hist.history[self.objective])
            return (best_score,)

        toolbox.register("individual", individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", objective_fn)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=self.pop_size)
        print(f"INITIAL POPULATIONS: {population}\n")

        NGEN = self.gen
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                print(f"GENERATION:{gen}, HP:{ind}, SCORE:{fit}\n")
                ind.fitness.values = fit
            print(
                f"GENERATION {gen} COMPLETED, BEST_HP: {tools.selBest(population, k=1)}\n\n"
            )
            population = toolbox.select(offspring, k=len(population))

        print(f"FINAL HP: {tools.selBest(population, k=len(population))}")
        self.best_pop = tools.selBest(population, k=1)

    def get_best_hyperparameters(self):
        return self.best_pop

    def build(self, params):
        model = self.model_builder(params)
        return model


class Hparams:
    def __init__(self) -> None:
        self.tools = base.Toolbox()

    def Int(self, param_name, min_value, max_value, step):
        self.tools.register(param_name, random.randrange, min_value, max_value, step)
        return self.tools.__dict__[param_name]

    def Choice(self, param_name, values):
        self.tools.register(param_name, random.choice, values)
        return self.tools.__dict__[param_name]
