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
        self.appeal:int = random.randint(0,100)

    def evaluate(self, evaluator:Evaluator):
        self.fitness = evaluator.evaluate(self.genes)

    def __str__(self):

        genes = ''.join(map(str,self.genes))
        return f"{self.sex} {self.fitness:<5.2f} {self.appeal:<5.2f} {genes} "

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
        sum_appeal = sum(ind.appeal for ind in self.inds)

        return {"max_fitness": max(ind.fitness for ind in self.inds),
                "min_fitness": min(ind.fitness for ind in self.inds),
                "avg_fitness": sum_fitness / self.size if self.size > 0 else float('inf'),
                "max_appeal": max(ind.appeal for ind in self.inds),
                "min_appeal": min(ind.appeal for ind in self.inds),
                "avg_appeal": sum_appeal / self.size if self.size > 0 else float('inf'),
                "num_males": sum(ind.sex for ind in self.inds)}

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
        result.appeal = random.choice([a.appeal, b.appeal])
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

    def __init__(self, crossover, cohort_size: int, mating_criterion: str):
        super().__init__(crossover)
        self.cohort_size = cohort_size
        self.mating_criterion = mating_criterion

    def pick_mate(self, input:Population, female:int, males: list[int])->int:
        """Returns the index of the selected mate."""
        mate = -1
        if self.mating_criterion == "max_appeal":
            mate = max(males, key=lambda idx: input[idx].appeal)
        elif self.mating_criterion == "closest_appeal":
            mate =  min(males, key=lambda idx: abs(input[idx].appeal - input[female].appeal))
        return mate

    def reproduce(self, input:Population, offspring: int)->Population:
        """Produce a new population."""
        result = Population(offspring, input.length)

        # the females will pick one male to mate amongst k candidates
        # possible variations: pick the one with more appeal
        # or with appeal closest to the female's appeal
        males: list[int] = [i for i, ind in enumerate(input.inds) if ind.sex==1]
        females: list[int] = [i for i, ind in enumerate(input.inds) if ind.sex==0]

        # need to iterate this multiple times until we get enough offspring
        current_offspring = 0
        for i in range(int(offspring/len(females)) + 1):
            self.mate_females(input, males, females, result, current_offspring, offspring)
            # current_offspring += len(females) this is incremented inside the func

        return result


    def mate_females(self, input, males, females, result, current_offspring, offspring):
        # the females will pick one mate amongst k candidates. we will pick
        # k males without replacement and reshuffle the list of males if needed

        # Track the current position in the male_indices list
        current_position = 0

        for female in females:
            if current_offspring == offspring:
                break

            # If there aren't enough males left in the list to pick k, reshuffle the males
            if current_position + self.cohort_size > len(males):
                random.shuffle(males)
                current_position = 0

            # Select k males from the current position
            selected_males = males[current_position:current_position + self.cohort_size]

            # Send the selected males and current female to the do_something function
            mate = self.pick_mate(input, female, selected_males)
            result[current_offspring] = self.crossover.crossover(input[female], input[mate])
            current_offspring += 1

            # Move the position forward by k
            current_position += self.cohort_size



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
        print(self.population.stats())

        for generation in range(0,20):
            selected = self.selection.select(self.population)
            self.population = self.reproduce.reproduce(selected, self.popsize)
            self.population.evaluate(self.evaluator)
            print(self.population.stats())
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