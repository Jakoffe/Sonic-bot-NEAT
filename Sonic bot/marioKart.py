import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")
imgarray = []
xpos_end = 0

# resume = False #Set to true if you want to continue training on the restore_file.
# restore_file = "neat-checkpoint-601" #Set this to the neweste training checkpoint.

def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        lives = 3
        xpos_max = 0
        oldScore = 0
        
        done = False


        while not done:
            
            env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)

            xpos = info['x']
            
            if xpos >= 65664:
                fitness_current += 10000000
                done = True
            
            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            if rew != 0:
                fitness_current += rew
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if info['lives'] != lives:
                lives = info['lives']
                current_max_fitness = int(current_max_fitness * 0.8)
                fitness_current = int(fitness_current * 0.8)
                
            if done or counter == 250:
                done = True
                print("Genome_id: " + str(genome_id) + "      Fitness: " + str(fitness_current))
                
            genome.fitness = fitness_current
    
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)