from eval_env import FileEnv

class RLEnv:
    def __init__(self, filename, proof_name, opts):
        self.opts = opts
        self.file_env = FileEnv(filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer=with_hammer, hammer_timeout=hammer_timeout)
        self.proof_env = None
        for proof_env in self.file_env:  # start a proof
            if proof_env.proof['name'] == proof_name:
                self.proof_env = proof_env
        #TODO: implement assert

    def give_reward(self, tac):
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
