class RLAgent:
    # TODO: implement
    def __init__(self):
        pass


    def prove_DFS(self, tac_template):
        # initialize the stack
        local_context, goal = parse_goal(obs['fg_goals'][0])
        tactics = self.model.beam_search(env, local_context, goal)
        stack = [[tac_template % tac.to_tokens() for tac in tactics[::-1]]]

