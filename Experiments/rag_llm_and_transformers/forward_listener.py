###########################################################################################
## This is just a quick hacked-up solution to intercept generated results from models.
## Probably not a good idea to use in practice, but to each their own.
## We're just using it for illustrating some examples.

import torch

class MethodListener:

    instances = []

    def __init__(self, obj, fn_name, listen_ins=True, listen_out=False, **kwargs):
        kwargs.update(dict(obj = obj, fn_name=fn_name, listen_ins=listen_ins, listen_out=listen_out))
        for k,v in kwargs.items(): setattr(self, k, v)
        self.fn = getattr(obj, fn_name, None)
        assert self.fn, f'{obj} did not have method {fn_name}'
        self.fn = getattr(self.fn, 'fn', self.fn)  ## avoid nesting
        self.__class__.instances += [self]

    def __call__(self, *args, **kwargs):
        comp_str = f"{getattr(self, 'name', 'obj')} {self.fn_name}"
        if self.listen_ins: print(f" - Inputs to {comp_str} = {self.arg_str_ins(*args, **kwargs)}")
        out = self.fn(*args, **kwargs)
        if self.listen_out: print(f" - Outputs of {comp_str} = {self.arg_str_out(out)}")
        return out

    def arg_str_ins(self, *args, **kwargs): return f"{args}, {kwargs}"
    def arg_str_out(self, *args, **kwargs): return f"{args}, {kwargs}"

    @classmethod
    def clear_all(cls): [listener.clear() for listener in cls.instances]
    def clear(self):    setattr(self.obj, self.fn_name, self.fn)
    def __del__(self):  self.clear()
    
    @classmethod
    @property
    def listen_ins(cls, state):
        for inst in cls.instances: inst.listen_ins = state
    
class ForwardListener(MethodListener):

    ignore_keys = {
        'attention_mask', 'inputs_embeds', 'head_mask', 'cross_attn_head_mask',  ## Not important during inference
        'use_cache', 'output_hidden_states', 'return_dict', 'output_attentions',  ## Boolean flags
        # 'past_key_values',               ## Very Important; shows states that persist through the decoding process
        'encoder_hidden_states', 'encoder_attention_mask'  ## not that important. Feel free to uncomment though
    }

    def __init__(self, *args, fn_name='forward', **kwargs):
        super().__init__(*args, fn_name=fn_name, **kwargs)

    def arg_str_ins(self, *args, **kwargs):
        return {k:self._parse_pair(k,v) for k,v in kwargs.items() if k not in ForwardListener.ignore_keys}

    def _parse_pair(self, k, v):
        if k == 'input_ids': return self.tokenizer.decode(v[0])
        if hasattr(v, 'shape'): return v.shape
        if type(v) in (list, tuple):
            try: return torch.stack(list(v)).shape
            except: return type(v)(self._parse_pair('', v1) for v1 in v)
        return v
    
class GenerateListener(MethodListener):

    def __init__(self, *args, fn_name='generate', **kwargs):
        super().__init__(*args, fn_name=fn_name, **kwargs)

    def arg_str_ins(self, *args, **kwargs):
        return f"\n{self.tokenizer.decode(kwargs.get('input_ids', [[]])[0])}"
