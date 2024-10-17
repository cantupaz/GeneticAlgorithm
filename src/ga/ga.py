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
    def __init__(self, popsize:int, length:int) -> None:
        self.size = popsize
        self.length = length
        self.inds:list[Individual] = [Individual(length) for i in range(popsize)]

    def evaluate(self, evaluator:Evaluator) -> None:
        for i in range(self.size):
            self.inds[i].evaluate(evaluator)

    def stats(self):

        sum_fitness = sum(ind.fitness for ind in self.inds)

        return {"max": max(self.inds, key=lambda x: x.fitness),
                "min": min(self.inds, key=lambda x: x.fitness),
                "avg": sum_fitness / self.size if self.size > 0 else float('inf')}

    def print(self):
        for i in range(self.size):
            print(i, self.inds[i])


class Selection:
    def __init__(self):
        pass

    def select(self, source:Population, destination:Population) -> None:
        pass

class TournamentSelection(Selection):
    def __init__(self, k:int=2):
        self.tournament_size = k

    def select(self, source:Population, destination:Population)->None:

        dest_index: int = 0
        indices: list[int] = list(range(source.size))
        for _ in range(self.tournament_size):
            random.shuffle(indices)

            # select k consecutive indices and determine the winner
            for i in range(0,len(indices) - self.tournament_size + 1, self.tournament_size):
                selected_indices = indices[i:i + self.tournament_size]

                # Determine the winner based on the fitness values
                winner = max(selected_indices, key=lambda idx: source.inds[idx].fitness)

                destination.inds[dest_index] = source.inds[winner]
                dest_index += 1



class MultiPointCrossover:
    def __init__(self, num_crossover_points:int=2, crossover_rate = 1.0):
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

    def reproduce(self, input:Population)->Population:
        """Produce a new population."""
        result = Population(input.size, input.length)
        indices: list[int] = list(range(input.size))


        dest_index:int = 0
        for _ in range(0,2):
            random.shuffle(indices)
            for i in range(0,result.size,2):
                result.inds[dest_index] = self.crossover.crossover(input.inds[indices[i]], input.inds[indices[i+1]])

        return result

class SexualReproduction(Reproduction):

    def reproduce(self, input:Population)->Population:
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
        selected = Population(self.popsize, self.length)

        self.population.evaluate(self.evaluator)
        stats = self.population.stats()
        print(f"avg {stats['avg']} max {stats['max'].fitness} min {stats['min'].fitness}")

        for generation in range(0,5):
            self.selection.select(self.population, selected)
            stats = selected.stats()
            print(f"selected avg {stats['avg']} max {stats['max'].fitness} min {stats['min'].fitness}")

            self.population = self.reproduce.reproduce(selected)
            self.population.evaluate(self.evaluator)
            stats = self.population.stats()
            print(f"new   avg {stats['avg']} max {stats['max'].fitness} min {stats['min'].fitness}")
            self.population.print()





def main():
    config = {}
    config["popsize"] = 10
    config["length"] = 20
    config["evaluator"] = OneMax() #TrapK(k=4)
    config["selection"] = TournamentSelection(k=2)
    config["reproduction"] = Reproduction(MultiPointCrossover(num_crossover_points=2, crossover_rate=1.0))

    ga = GeneticAlgorithm(config)
    ga.run()

if __name__ == '__main__':
    main()