from sys import maxsize
from random import random, randint, sample
from time import time
from flatland.envs.rail_env import RailEnv
from search import search

"""
code modified from :
https://github.com/lccasagrande/TSP-GA/blob/master/src/tsp_ga.py
"""

env = None #global

class Gene:
    """
    Represent Agent node
    """

    def __init__(self, id):
        self.agent_id = id


def build_agents_nodes(env):
    """
    Args:
        env: Rail environment

    Returns: agent nodes

    """
    agents_node = []
    for i in env.get_agent_handles():
        agent_node = Gene(i)
        agents_node.append(agent_node)

    return agents_node


class Individual :
    """
    Represent a set of permutations of agent scheduling order.
    Each Indiviudal consists of a set of genes that represent each agent
    """
    def __init__(self,genes):
        self.genes = genes
        self.g_reward = 0
        self.fitness = 0

    def swap(self,gene,other_gene):
        a, b = self.genes.index(gene), self.genes.index(other_gene)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.g_reward = 0
        self.fitness = 0

    def add(self,gene):
        self.genes.append(gene)
        self.g_reward = 0
        self.fitness = 0

    def get_fitness(self):
        """
        Get the fitness value of each Individuals based on the g value A* computed for the
        Individual (agent order)

        Returns: fitness score

        """
        _, self.g_reward = search(self.genes, env)
        self.fitness = 10/self.g_reward
        return self.fitness


class Population:
    """
    Represent a set of combinations of permutations of agent scheduling order.
    Each Population consists of a set of Individual.
    """

    def __init__(self, individual):
        self.individuals = individual

    @staticmethod
    def gen_individuals(sz, genes):
        """
        Args:
            sz: size of of Pupulation
            genes: set of agent node used to compute Individual

        Returns:
            generated Population
        """

        individuals = []
        for _ in range(sz):
            individuals.append(Individual(sample(genes, len(genes))))

        return Population(individuals)

    def add(self,order):
        self.individuals.append(order)

    def rmv(self, order):
        self.individuals.remove(order)

    def get_fittest(self):
        fittest = self.individuals[0]
        for i in range(len(self.individuals)):
            fitness = self.individuals[i].get_fitness()

            if fitness > fittest.fitness:
                fittest = self.individuals[i]

        return fittest

def evolve(pop, mut_rate):
    new_generation = Population([])
    pop_size = len(pop.individuals)
    elitism_num = pop_size//2

    # Elitism
    for _ in range(elitism_num):
        fittest = pop.get_fittest()
        new_generation.add(fittest)
        pop.rmv(fittest)

    # Crossover
    for _ in range(elitism_num, pop_size):
        parent_1 = selection(new_generation)
        parent_2 = selection(new_generation)
        child = crossover(parent_1, parent_2)
        new_generation.add(child)

    # Mutation
    for i in range(elitism_num, pop_size):
        mutation(new_generation.individuals[i], mut_rate)

    return new_generation


def crossover(parent_1, parent_2):
    def fill_with_parent1_genes(child, parent, genes_n):
        start_at = randint(0, len(parent.genes) - genes_n - 1)
        finish_at = start_at + genes_n
        for i in range(start_at, finish_at):
            child.genes[i] = parent_1.genes[i]

    def fill_with_parent2_genes(child, parent):
        j = 0
        for i in range(0, len(parent.genes)):
            if child.genes[i] == None:
                while parent.genes[j] in child.genes:
                    j += 1
                child.genes[i] = parent.genes[j]
                j += 1

    genes_n = len(parent_1.genes)
    child = Individual([None for _ in range(genes_n)])
    fill_with_parent1_genes(child, parent_1, genes_n // 2)
    fill_with_parent2_genes(child, parent_2)

    return child


def mutation(individual, rate):
    for _ in range(len(individual.genes)):
        if random() < rate:
            sel_genes = sample(individual.genes, 2)
            individual.swap(sel_genes[0], sel_genes[1])


def selection(population):
    return Population(sample(population.individuals, 2)).get_fittest()


def run_ga(environment :RailEnv):
    global env
    env = environment

    if env.number_of_agents > 7:
        n_gen = 5
        mut_rate = 0.15
        agent_order = build_agents_nodes(env)
        population = Population.gen_individuals(5, agent_order)
    else:
        n_gen = 10
        mut_rate = 0.1
        agent_order = build_agents_nodes(env)
        population = Population.gen_individuals(10, agent_order)

    counter, generations, min_cost = 0, 0, maxsize


    while counter < n_gen:
        population = evolve(population,mut_rate)
        cost = population.get_fittest().g_reward
        if cost > min_cost:
            counter, min_cost = 0, cost
        else:
            counter += 1

        generations += 1


    order = population.get_fittest()

    return order.genes

