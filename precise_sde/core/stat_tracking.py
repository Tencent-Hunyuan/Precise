import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo', sentinel=-10.0):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0

        # mask: per-sample bool, True = valid (no sentinel in any timestep)
        if rewards.ndim == 1:
            valid_mask = rewards != sentinel
        else:
            valid_mask = np.all(rewards != sentinel, axis=1)

        for prompt in unique:
            prompt_idx = prompts == prompt
            prompt_valid = valid_mask[prompt_idx]
            prompt_rewards = rewards[prompt_idx]
            valid_rewards = prompt_rewards[prompt_valid]
            if prompt not in self.stats:
                self.stats[prompt] = []
            if len(valid_rewards) > 0:
                self.stats[prompt].extend(valid_rewards)
            self.history_prompts.add(hash(prompt))
        for prompt in unique:
            prompt_idx = prompts == prompt
            prompt_valid = valid_mask[prompt_idx]
            if not np.any(prompt_valid) or len(self.stats.get(prompt, [])) == 0:
                advantages[prompt_idx] = 0.0
                continue
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompt_idx]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                valid_global = rewards[valid_mask] if np.any(valid_mask) else rewards
                std = np.std(valid_global, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                adv = (prompt_rewards - mean) / std
                adv[~prompt_valid] = 0.0
                advantages[prompt_idx] = adv
            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()