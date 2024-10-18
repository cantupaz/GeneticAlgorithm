# Simple Genetic Algorithm

A simple GA in Python with a form of sexual selection. Genetic algorithms are simple search and optimization algorithms based on ideas of evolution by natural selection. GAs maintain a population of candidate solutions and in each iteration they select promising solutions using a fitness function, the selected individuals mate randomly and recombine their genes to produce the next generation of solutions. The individuals in a simple GA do not have sexes and mates are chosen randomly. In contrast, in Nature many species have sexes and mating is not random. Sexual selection is about members of one sex competing to gain acces to members of the other sex. There are very few studies about sexual selection in GAs and it is unclear what advantages they may have over the simpler panmictic random selection.

The genetic algorithm here includes a form of sexual selection in addition to the standard fitness-based selection. The standard selection algorithms used in GAs can be seen as "these individuals are fit to survive." We leave those algorithms unchanged as they provide the pressure to improve fitness. The sexual selection is introduced by assigning individuals a sex and letting the females select a mate from a cohort of males. We also assign individuals an "appeal" score and females can choose their mates either by choosing the male with the highest appeal score in the cohort or the male whose appeal score more closely resembles the score of the female.

Preliminary results suggest that, for the same effort, the GAs with sexual selection reaches solutions of slightly lower quality than the standard GA. However, the appeal score is also improved when using sexual selection. This suggests that sexual selection can be used as a simple multi-objective optimizer.

Stuff to read:

- When Selection Meets Seduction, ICGA 95
- several others

---
