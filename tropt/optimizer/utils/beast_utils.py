import torch


@torch.no_grad()
def sample_top_p(probs, p, return_tokens=0):
    """
    Masks out the bottom (1-p) fraction from token probabilities,
    and returns the next_token / all probability indices.
    Params:
        probs: softmax logit values
        p: top_p
        return_tokens: no. of tokens returned
    Return:
        next_token: set of next tokens
    """
    # If probs do not sum to (roughly) 1, apply softmax
    if not torch.allclose(
        probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-3
    ):
        probs = torch.softmax(probs, dim=-1)

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    if torch.all(probs_sort.sum(dim=-1, keepdim=True) == 0):
        # if all probabilities are zero, this means that the top_p is too low.
        # in this case, we will just use the top_k probabilities.
        # get the top 1 token
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sort[:, 1:] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=max(1, return_tokens))
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
