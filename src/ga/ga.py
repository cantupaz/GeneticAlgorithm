"""Implement a simple GA with sexual selection."""

from typing import Any
import random


class Evaluator:
    def evaluate(self, genes:list[int])->float:
        return 0.0

class TrapK(Evaluator):
    def __init__(self, k:int=4):
        self.k = k

    def evaluate(self, genes:list[int]) -> float:

        k = self.k
        assert(len(genes) % k == 0)

        eval:int = 0
        for i in range(0, len(genes), k):
            count = sum(genes[i:i+k])
            if count == k:
                trap = k
            else:
                trap = k-1-count
            eval += trap

        return eval

class OneMax(Evaluator):
    def evaluate(self, genes:list[int])->float:
        return sum(genes)

class Individual:
    def __init__(self, length:int) -> None:
        self.length = length
        self.genes:list[int] = [random.randint(0,1) for i in range(length)]
        self.fitness:float = 0.0
        self.sex:int = random.randint(0,1)
        self.attractiveness:int = random.randint(0,100)

    def evaluate(self, evaluator:Evaluator):
        self.fitness = evaluator.evaluate(self.genes)

    def __str__(self):

        genes = ''.join(map(str,self.genes))
        return f"{self.sex} {self.fitness:<5.2f} {self.attractiveness:<5.2f} {genes} "

class Population:
    """Holds a population."""
    def __init__(self, popsize:int, length:int) -> None:
        self.size = popsize
        self.length = length
        self.inds:list[Individual] = [Individual(length) for i in range(popsize)]

    def evaluate(self, evaluator:Evaluator) -> None:
        """Evaluate every individual in the population."""
        for i in range(self.size):
            self.inds[i].evaluate(evaluator)

    def stats(self):
        """Return a dict with statistics about the population (min, max, avg fitness)."""
        sum_fitness = sum(ind.fitness for ind in self.inds)

        return {"max": max(self.inds, key=lambda x: x.fitness),
                "min": min(self.inds, key=lambda x: x.fitness),
                "avg": sum_fitness / self.size if self.size > 0 else float('inf')}

    def __getitem__(self, index:int)->Individual:
        """Returns the ith individual."""
        return self.inds[index]

    def __setitem__(self, index:int, value:Individual):
        """Set the i-th individual."""
        self.inds[index] = value

    def print(self):
        for i in range(self.size):
            print(i, self.inds[i])


class Selection:
    def __init__(self):
        pass

    def select(self, source:Population)->Population:
        pass

class TournamentSelection(Selection):
    def __init__(self, k:int=2):
        self.tournament_size = k

    def select(self, source:Population) ->Population:

        destination = Population(source.size, source.length)
        dest_index: int = 0
        indices: list[int] = list(range(source.size))
        for _ in range(self.tournament_size):
            random.shuffle(indices)

            # select k consecutive indices and determine the winner
            for i in range(0,len(indices) - self.tournament_size + 1, self.tournament_size):
                selected_indices = indices[i:i + self.tournament_size]

                # Determine the winner based on the fitness values
                winner = max(selected_indices, key=lambda idx: source[idx].fitness)

                destination[dest_index] = source[winner]
                dest_index += 1
        return destination

class TruncationSelection(Selection):
    def __init__(self, fraction: float, property: str="fitness"):
        self.fraction: float = fraction
        self.property:str = property

    def select(self, source:Population)->Population:
        """Return a population with the top individuals based on the factor indicated."""

        dest_size = int(source.size * self.fraction)
        destination = Population(dest_size, source.length)
        source.inds = sorted(source.inds, key=lambda ind: getattr(ind, self.property), reverse=True) # desc order
        for i in range(dest_size):
            destination[i] = source[i]
        return destination



class MultiPointCrossover:
    def __init__(self, num_crossover_points:int=2, crossover_rate:float = 1.0):
        self.num_crossover_points = num_crossover_points
        self.crossover_rate: float = crossover_rate


    def crossover(self, a: Individual, b: Individual) -> Individual:
        """Crossover two individuals."""

        if random.random() > self.crossover_rate:
            return a

        # set up the offspring
        result = Individual(a.length)
        result.sex = random.randint(0,1)
        result.attractiveness = random.choice([a.attractiveness, b.attractiveness])
        result.genes = []

        # Generate unique random crossover points
        crossover_points = sorted(random.sample(range(1, a.length), self.num_crossover_points))

        # crossover
        use_a = random.choice([True, False])
        start = 0
        for point in crossover_points:
            if use_a:
                result.genes.extend(a.genes[start:point])
            else:
                result.genes.extend(b.genes[start:point])

            use_a = not use_a
            start = point

        # After the last crossover point, append the remaining part
        if use_a:
            result.genes.extend(a.genes[start:])
        else:
            result.genes.extend(b.genes[start:])


        return result



class Reproduction:
    def __init__(self, crossover: MultiPointCrossover):
        self.crossover = crossover

    def reproduce(self, input:Population, offspring: int)->Population:
        """Produce a new population with the number of offspring indicated. The input population does not
        have to be of the same size as the required offspring."""
        result = Population(offspring, input.length)

        # create a randomly shuffled list of parents
        indices: list[int] = list(range(input.size))
        parents: list[int] = []
        while len(parents) < offspring*2:
            random.shuffle(indices)
            parents.extend(indices)
        parents = parents[:offspring*2]

        # mate two random parents
        for i in range(0, offspring*2, 2):
            result[int(i/2)] = self.crossover.crossover(input[parents[i]], input[parents[i+1]])

        return result

class SexualReproduction(Reproduction):

    def reproduce(self, input:Population, offspring: int)->Population:
        """Produce a new population."""
        result = Population(input.size, input.length)

        return result




class GeneticAlgorithm:

    def __init__(self, config:dict[str, Any]) -> None:

        self.popsize:int = config.get("popsize" ,100)
        self.length:int = config.get("length", 100)
        self.selection:Selection = config.get("selection", TournamentSelection())
        self.reproduce:Reproduction = config.get("reproduction", Reproduction(MultiPointCrossover()))
        self.evaluator: Evaluator = config.get("evaluator", TrapK())

        self.population:Population = Population(self.popsize, self.length)

    def run(self):
        # selected = Population(self.popsize, self.length)

        self.population.evaluate(self.evaluator)
        stats = self.population.stats()
        print(f"avg {stats['avg']} max {stats['max'].fitness} min {stats['min'].fitness}")

        for generation in range(0,20):
            selected = self.selection.select(self.population)
            self.population = self.reproduce.reproduce(selected, self.popsize)
            self.population.evaluate(self.evaluator)
            stats = self.population.stats()
            print(f"new   avg {stats['avg']} max {stats['max'].fitness} min {stats['min'].fitness}")
            # self.population.print()





def main():
    config = {}
    config["popsize"] = 20
    config["length"] = 20
    config["evaluator"] = OneMax() #TrapK(k=4)
    config["selection"] = TruncationSelection(0.5) #TournamentSelection(k=2)
    config["reproduction"] = Reproduction(MultiPointCrossover(num_crossover_points=2, crossover_rate=1.0))

    ga = GeneticAlgorithm(config)
    ga.run()

if __name__ == '__main__':
    main()