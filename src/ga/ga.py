from typing import Any
import random

class Individual:
    def __init__(self, length:int, eval) -> None:
        self.genes:list[int] = [random.randint(0,1) for i in range(length)]
        self.fitness = eval
        self.sex:int = random.randint(0,1)
        self.attractiveness:int = random.randint(0,100)


class Population:
    def __init__(self, popsize:int, length:int) -> None:
        self.size = popsize

        self.inds:list[Individual] = [Individual(length) for i in range(popsize)]

class Selection:
    def __init__(self):
        pass

    def select(self, source:Population) -> Population:
        pass

class TournamentSelection(Selection):
    def __init__(self, k:int=2):
        self.tournament_size = k

    def select(self, source:Population, destination:Population):

        dest_index: int = 0
        for _ in range(self.tournament_size):
            indices: list[int] = list(range(source.size))
            random.shuffle(indices)

            # select k consecutive indices and determine the winner
            for i in range(len(indices) - k + 1):  # Ensure we have enough remaining indices
                selected_indices = indices[i:i + k]

                # Determine the winner based on the fitness values
                winner = max(selected_indices, key=lambda idx: source.inds[idx].fitness)

                destination[dest_index] = source.inds[winner]
                dest_index += 1

class MultiPointCrossover:
    def __init__(self, num_crossover_points:int=2):
        self.num_crossover_points = num_crossover_points


    def crossover(self, a: Individual, b: Individual) -> Individual:

        # Generate unique random crossover points
        crossover_points = sorted(random.sample(range(1, a.length), self.num_crossover_points))

        result = a.copy()
        use_a = True
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



class GeneticAlgorithm:

    def __init__(self, config:dict[str, Any]) -> None:

        self.popsize:int = config.get("popsize")
        self.length:int = config.get("length")
        self.selection = config.set("selection")
        self.crossover_rate:float = config.get("crossover rate",1.0)
        self.crossover = config.get("crossover")

        self.population = Population(self.popsize, self.length)

    def run():
        for generation in range(0,100):
            selected = self.selection.select()




def trap_k(genes:list[int], k:int) -> int:

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



def main():
    config = {}
    config["popsize"] = 100
    config["length"] = 100
    config["eval"] = trap_k(4)
    config["select"] = TournamentSelection(k=2)
    config["crossover"] = MultiPointCrossover(num_crossover_points=2)
    config["crossover rate"] = 1.0

    ga = GeneticAlgorithm(config)
    ga.run()

if __name__ == '__main__':
    main()