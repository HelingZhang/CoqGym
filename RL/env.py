from eval_env import FileEnv
import glob
import serapi

class RLEnv:
    def __init__(self, filename, proof_name):
        self.file_env = FileEnv(filename, 500, 1000)
        self.proof_env = None
        for proof_env in self.file_env:  # start a proof
            if proof_env.proof['name'] == proof_name:
                self.proof_env = proof_env
        #TODO: implement assert

    def step(self, tac):
        # Gives reword to a tactic given by RL agent.
        reward = 0
        obs = self.proof_env.init()
        print_goals(obs)
        obs = self.proof_env.step(tac)
        print(obs['result'])
        print_goals(obs)
        if obs['result'] == 'SUCCESS':
            reward = 1
        elif obs['result'] == 'PROVING':
            reward = 0
        else:
            reward = -1

        return reward

# printing functions.
def print_single_goal(g):
    for h in g['hypotheses']:
        for ident in h['idents']:
            print('\t%s: %s' % (ident, h['type']))
    print('---------------')
    print('\t%s' % g['type'])
    print('##########')


def print_goals(obs):
    if 'fg_goals' not in obs:
        print('##########')
        return
    print('########## fg_goals ##########')
    for g in obs['fg_goals']:
        print_single_goal(g)
    print('########## bg_goals ##########')
    for g in obs['bg_goals']:
        print_single_goal(g)
    print('########## shelved_goals ##########')
    for g in obs['shelved_goals']:
        print_single_goal(g)
    print('########## given_up_goals ##########')
    for g in obs['given_up_goals']:
        print_single_goal(g)

if __name__=="__main__":
    import time
    import statistics
    import pickle
    # benchmark the RL environment

    files = [f for f in glob.glob("../data/*/*.json")]

    # record the time it takes for a step in the RL env.
    step_time = []

    for f in files:
        # extract proof names & proof steps (tacs)
        file_env = FileEnv(f, max_num_tactics=100, timeout=600)
        names = [pf['name'] for pf in file_env.proofs]
        tacs = [
            [step['command'][0] for step in pf['steps']] for pf in file_env.proofs
        ]
        for (i, name) in enumerate(names):
            # extract tacs for each proof
            try:
                env = RLEnv(f, name)
            except serapi.CoqExn:
                print("CoqExn")
                continue
            for tac in tacs[i]:
                start = time.time()
                try:
                    reword = env.step(tac)
                except:
                    break
                end = time.time()
                step_time.append(end - start)
                print("Step # {}, time = {}".format(len(step_time), end-start))
        print("change file")

    avg_step_time = sum(step_time) / len(step_time)
    std_step_time = statistics.stdev(step_time)
    max_step_time = max(step_time)
    print('finished')

    with open("step_time.pickle", 'wb') as fp:
        pickle.dump(step_time, fp)

    # f = '../data/StructTact/Assoc.json'
    #
    # step_time = []
    #
    # file_env = FileEnv(f, max_num_tactics=100, timeout=600)
    # names = [pf['name'] for pf in file_env.proofs]
    # tacs = [
    #     [step['command'][0] for step in pf['steps']] for pf in file_env.proofs
    # ]
    # for (i, name) in enumerate(names):
    #     # extract tacs for each proof
    #     try:
    #         env = RLEnv(f, name)
    #     except serapi.CoqExn:
    #         print("CoqExn")
    #         continue
    #     for tac in tacs[i]:
    #         start = time.time()
    #         reword = env.step(tac)
    #         end = time.time()
    #         step_time.append(end - start)
    #         print("Step # {}, time = {}".format(len(step_time), end-start))
    #
    # avg_step_time = sum(step_time)/len(step_time)
    # std_step_time = statistics.stdev(step_time)
    # max_step_time = max(step_time)
    # print("change file")
