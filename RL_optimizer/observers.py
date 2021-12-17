from tf_agents.trajectories import StepType, Trajectory

class RewardAccumulator:
    def __init__(self, store_steps: int = False):
        self.store_steps = store_steps
        self.reset()

    def __call__(self, trajectory: Trajectory):
        trajectory_reward = trajectory.reward.numpy()[0]

        if trajectory.is_first():
            self.discount = 1. # reset discount

            self.rewards.append(
                [trajectory_reward] if self.store_steps
                else trajectory_reward
            )
        elif trajectory.is_last():
            # note that environment returns one extra step post-termination (bug)
            # workaround is provided instead (this switch statement should not exist)
            pass
        elif self.store_steps:
            self.rewards[-1].append(
                self.rewards[-1][-1] + self.discount * trajectory_reward
            )
        else: # store sum only
            self.rewards[-1] += self.discount * trajectory_reward
        
        self.discount *= trajectory.discount.numpy()[0]

    def reset(self):
        self.rewards = []
        self.discount = 1.

if __name__ == "__main__":
    pass