from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch


def reward_func(prompt, response):
    """Reward function that gives higher scores to longer completions."""
    return float(len(response))


class CustomRewardManager:
    """The custom reward manager.
    在这个自定义的类中，有一个for循环用于计算每个对应的response的reward值，
    由于原始代码中只给出了prompt_id和response_id，
    因此，如果自定义的奖励函数需传入str，需要先decode得到字符串，然后使用reward_func得到奖励即可。
    例如，可以直接将response的长度当做奖励，来鼓励模型输出更长的回复：
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # data.batch keys:
        # 1. responses: response tokens
        # 2. prompts: 

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode reward computer需要str的话
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # custom score
            prompt = self.tokenizer.decode(valid_prompt_ids)
            response = self.tokenizer.decode(valid_response_ids)
            # 这里可以写你自定义的reward函数
            score = reward_func(prompt=prompt, response=response)
            
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
