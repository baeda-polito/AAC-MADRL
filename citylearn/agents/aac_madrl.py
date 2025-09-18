from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from pathlib import Path
from typing import Optional
import json, zipfile, tempfile
import numpy as np
import torch
import numpy as np
import numpy.typing as npt
import re
from itertools import chain

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rlc import RLC
from citylearn.agents.rbc import RBC
from citylearn.rl import ReplayBuffer


MSELoss = torch.nn.MSELoss()
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True
def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    #acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acs = torch.zeros(*probs.shape, device=device).scatter_(1, int_acs, 1)
    return int_acs, acs

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        # concatenate state and action for each agent
        inps = [torch.cat([states[a_i]] + actions[a_i], dim=1) for a_i in agents]    
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = [action_tensor.max(dim=1, keepdim=True)[1] for action_tensor in actions[a_i]]
            int_acs = torch.cat(int_acs, dim=1)

            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, num_classes, sample, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=False, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.num_classes = num_classes
        self.sample = sample

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out

class MultiDiscretePolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super(MultiDiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, num_classes, sample, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):

        out = super(MultiDiscretePolicy, self).forward(obs)
        num_classes = [c for c in num_classes if c != 0]
        splits = torch.split(out, num_classes, dim=1)  
        probs = [F.softmax(split, dim=1) for split in splits]  
        
        on_gpu = next(self.parameters()).is_cuda

        if sample:
            int_acts = []
            acts = []
            for prob in probs:
                int_act, act = categorical_sample(prob, use_cuda=on_gpu)
                int_acts.append(int_act)
                acts.append(act)
        else: ##DEPLOYMENT
            acts = [onehot_from_logits(split) for split in splits]
            int_act = [torch.argmax(a, dim=1) for a in acts]
            int_acts = [t.view(t.shape[0], 1) for t in int_act]

        rets = [acts]
        if return_log_pi or return_entropy:
            log_probs = [F.log_softmax(split, dim=1) for split in splits]
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(
                torch.cat([log_prob.gather(1, int_act) for log_prob, int_act in zip(log_probs, int_acts)], dim=1))
        if regularize:
            rets.append([(out ** 2).mean()])
        if return_entropy:
            rets.append(
                -(torch.cat([log_prob * prob for log_prob, prob in zip(log_probs, probs)], dim=1)).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets

class AAC_MADRL(RLC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        r"""Actor-Attention algorithm.

        Parameters
        ----------
        env: CityLearnEnv
            CityLearn environment.

        Other Parameters
        ----------------
        **kwargs : Any
            Other keyword arguments used to initialize super class.
        """

        super().__init__(env, **kwargs)

        classes = kwargs.pop('classes', {
            "cooling_storage": 21,
            "heating_storage": 21,
            "dhw_storage": 21,
            "electrical_storage": 21,
            "cooling_device": 11,
            "heating_device": 11,
            "cooling_or_heating_device": 21
        })

        attend_heads = kwargs.pop('attend_heads', 1)
        sample = kwargs.pop('sample', True)
        lr = kwargs.pop('lr', 3e-4)

        self.nagents = len(self.action_space)
        self.policy_net = [None] * self.nagents
        self.target_policy_net = [None] * self.nagents
        self.policy_optimizer = [None] * self.nagents
        self.action_dim = [None] * self.nagents
        self.lr = lr

        self.norm_mean = [None] * self.nagents
        self.norm_std = [None] * self.nagents
        self.r_norm_mean = [None] * self.nagents
        self.r_norm_std = [None] * self.nagents
        self.normalized = [False] * self.nagents

        self.replay_buffer = [ReplayBuffer(int(self.replay_buffer_capacity)) for _ in range(self.nagents)]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.niter = 0
        self.reward_scale = 10.0
        self.classes = classes
        self.attend_heads = attend_heads
        self.num_classes = list(self.classes.values())
        self.defined_actions = {key: np.linspace(0, 1, num) if key in ['cooling_device', 'heating_device'] else np.linspace(-1, 1, num) for key, num in self.classes.items()}
        
        self.sample = sample
        self.set_networks()

    def set_networks(self, internal_observation_count: int = None):

        internal_observation_count = 0 if internal_observation_count is None else internal_observation_count
        observation_dim = [self.observation_dimension[agent] + internal_observation_count for agent in range(self.nagents)]
        action_bool = [[1 if key in self.action_names[agent] else 0 for key in self.classes.keys()] for agent in range(self.nagents)]
        self.action_dim = [[n_clas * a_bool for n_clas, a_bool in zip(self.num_classes, action_bool[agent])] for agent in range(self.nagents)]
        action_dim_sum = [sum(self.action_dim[agent]) for agent in range(self.nagents)]
        sa_sizes = [[observation_dim[agent], action_dim_sum[agent]] for agent in range(self.nagents)]
        self.policy_net = [MultiDiscretePolicy(input_dim=observation_dim[agent], out_dim=action_dim_sum[agent], num_classes=self.action_dim[agent], sample=self.sample).to(self.device) for agent in range(self.nagents)]
        self.target_policy_net = [MultiDiscretePolicy(input_dim=observation_dim[agent], out_dim=action_dim_sum[agent], num_classes=self.action_dim[agent], sample=self.sample).to(self.device)  for agent in range(self.nagents)]
        self.policy_optimizer = [Adam(self.policy_net[agent].parameters(), lr=self.lr) for agent in range(self.nagents)]
        for agent in range(self.nagents):
            hard_update(self.target_policy_net[agent], self.policy_net[agent])
        self.critic = AttentionCritic(sa_sizes=sa_sizes, attend_heads=self.attend_heads).to(self.device) 
        self.target_critic = AttentionCritic(sa_sizes=sa_sizes, attend_heads=self.attend_heads).to(self.device) 
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-3)
    
    def save_models(self, zip_path: str = "maac.zip", dtype: Optional[str] = None) -> str:
        """
        Salva SOLO uno ZIP contenente:
        - mean_std/config.json (statistiche di normalizzazione)
        - policy_net_i.pt e target_policy_net_i.pt per ogni azione
        - critic.pt e target_critic.pt

        Parametri
        ---------
        zip_path : str
            Percorso del file .zip da creare.
        dtype : Optional[str]
            'fp16' o 'bf16' per ridurre dimensione (cast dei tensori floating).
            'fp32' o None per mantenere il dtype originale.
        """
        

        def _to_torch_dtype(d):
            if d is None: return None
            d = d.lower()
            if d in ("fp16", "float16"):   return torch.float16
            if d in ("bf16", "bfloat16"):  return torch.bfloat16
            if d in ("fp32", "float32"):   return torch.float32
            raise ValueError("dtype non supportato: %s" % d)

        tgt_dtype = _to_torch_dtype(dtype)

        print(".... saving MAAC models to ZIP ....")
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            # 1) mean/std -> JSON portabile
            ms_dir = out_dir / "mean_std"
            ms_dir.mkdir(parents=True, exist_ok=True)
            params = {
                "mean": [np.asarray(arr).tolist() for arr in getattr(self, "norm_mean", [])],
                "std":  [np.asarray(arr).tolist() for arr in getattr(self, "norm_std", [])],
            }
            (ms_dir / "config.json").write_text(json.dumps(params, indent=2))

            # 2) helper cast
            def _cast(sd):
                if tgt_dtype is None: 
                    return sd
                return {k: (v.to(tgt_dtype) if torch.is_floating_point(v) else v)
                        for k, v in sd.items()}

            # 3) policy + target (per azione)
            if hasattr(self, "policy_net"):
                n = len(self.policy_net)
            elif hasattr(self, "action_dimension"):
                n = len(self.action_dimension)
            else:
                raise RuntimeError("Impossibile determinare il numero di policy da salvare.")

            for i in range(n):
                # policy i
                torch.save(_cast(self.policy_net[i].state_dict()), out_dir / f"policy_net_{i}.pt")
                # target policy i
                torch.save(_cast(self.target_policy_net[i].state_dict()), out_dir / f"target_policy_net_{i}.pt")

            # 4) critici (centrali)
            torch.save(_cast(self.critic.state_dict()),         out_dir / "critic.pt")
            torch.save(_cast(self.target_critic.state_dict()),  out_dir / "target_critic.pt")

            # 5) crea zip finale
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in out_dir.rglob("*"):
                    if f.is_file(): 
                        zf.write(f, f.relative_to(out_dir))

        print(f".... AAC-MADRL models saved in ZIP: {zip_path} ....")
        return str(zip_path)
    """
    def save_models(self, directory: str = 'maac_param_net'):
        print('.... saving models ....')
        
        params = {"mean": [arr.tolist() for arr in self.norm_mean], "std": [arr.tolist() for arr in self.norm_std]}
        with open(f'{directory}\mean_std\config.py', 'w') as f:
            f.write(f'params = {params}\n')
        for i in range(len(self.action_dimension)):
            torch.save(self.policy_net[i].state_dict(), f'{directory}/policy_net_{i}.pt')
            torch.save(self.target_policy_net[i].state_dict(), f'{directory}/target_policy_net_{i}.pt')
        torch.save(self.critic.state_dict(), f'{directory}/critic.pt')
        torch.save(self.target_critic.state_dict(), f'{directory}/target_critic.pt')
        print('.... models saved ....')
    
    def load_models(self, directory: str = 'maac_param_net'):
        print('.... loading models ....')
                      
        for i in range(len(self.action_dimension)):
            self.policy_net[i].load_state_dict(torch.load(f'{directory}/policy_net_{i}.pt', map_location=torch.device('cpu')))
            self.target_policy_net[i].load_state_dict(torch.load(f'{directory}/target_policy_net_{i}.pt', map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(f'{directory}/critic.pt', map_location=torch.device('cpu')))
        self.target_critic.load_state_dict(torch.load(f'{directory}/target_critic.pt', map_location=torch.device('cpu')))
        print('.... models loaded ....')
    """

    def load_models(self,
                    zip_path: str,
                    map_location: Optional[str] = None,
                    cast_to: Optional[str] = None,
                    strict: bool = True) -> None:
        """
        Carica pesi e normalizzazioni da uno ZIP creato da save_models() di AAC-MADRL.

        Parametri
        ---------
        zip_path : str
            Percorso allo zip (es. '.../aac_madrl.zip' o '.../maac.zip').
        map_location : Optional[str]
            'cpu', 'cuda', o torch.device (passato a torch.load).
        cast_to : Optional[str]
            Se 'fp16'/'bf16'/'fp32', forza il cast dei tensori floating del checkpoint
            prima del load (utile se i pesi sono stati salvati in un dtype diverso).
            Se None, prova ad allineare automaticamente al dtype del modulo di destinazione.
        strict : bool
            Passato a module.load_state_dict. Metti False se ci sono key extra/mancanti.
        """
        import json, zipfile, tempfile
        from pathlib import Path
        import numpy as np
        import torch

        def _to_torch_dtype(d: Optional[str]):
            if d is None: return None
            d = d.lower()
            if d in ("fp16", "float16"):   return torch.float16
            if d in ("bf16", "bfloat16"):  return torch.bfloat16
            if d in ("fp32", "float32"):   return torch.float32
            raise ValueError("dtype non supportato: %s" % d)

        tgt_dtype = _to_torch_dtype(cast_to)

        zp = Path(zip_path)
        if not zp.exists():
            raise FileNotFoundError(f"Zip non trovato: {zp}")

        print(f".... loading AAC-MADRL from ZIP: {zp} ....")

        # Estrai lo zip in una cartella temporanea
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(td)
            tmp = Path(td)

            # 1) mean/std
            ms = tmp / "mean_std" / "config.json"
            if ms.exists():
                params = json.loads(ms.read_text())
                self.norm_mean = [np.asarray(x) for x in params.get("mean", [])]
                self.norm_std  = [np.asarray(x) for x in params.get("std",  [])]
            else:
                print("[warn] mean_std/config.json non trovato: salto normalizzazioni.")

            # 2) deduci quanti head/azioni caricare
            #    priorità: struttura dell'istanza → file presenti
            if hasattr(self, "policy_net"):
                n_heads = len(self.policy_net)
            elif hasattr(self, "action_dimension"):
                n_heads = len(self.action_dimension)
            else:
                # deduci dai file policy_net_<i>.pt
                rx = re.compile(r"policy_net_(\d+)\.pt")
                idx = [int(m.group(1)) for p in tmp.rglob("policy_net_*.pt") if (m := rx.fullmatch(p.name))]
                if not idx:
                    raise RuntimeError("Impossibile dedurre il numero di policy da caricare (manca policy_net_*.pt).")
                n_heads = max(idx) + 1

            # 3) helper cast/load
            def _cast_state(sd, module):
                # Se richiesto, forza il dtype target
                if tgt_dtype is not None:
                    return {k: (v.to(tgt_dtype) if torch.is_floating_point(v) else v) for k, v in sd.items()}
                # Altrimenti allinea al dtype del modulo
                try:
                    target_dtype = next(module.parameters()).dtype
                    return {k: (v.to(target_dtype) if torch.is_floating_point(v) and v.dtype != target_dtype else v)
                            for k, v in sd.items()}
                except StopIteration:
                    return sd  # modulo senza parametri

            def _load_into(module, ckpt_path: Path):
                sd = torch.load(ckpt_path, map_location=map_location)
                sd = _cast_state(sd, module)
                module.load_state_dict(sd, strict=strict)

            # 4) policy + target policy per head
            for i in range(n_heads):
                # policy
                p = tmp / f"policy_net_{i}.pt"
                if hasattr(self, "policy_net") and len(self.policy_net) > i and p.exists():
                    _load_into(self.policy_net[i], p)
                else:
                    if not p.exists():
                        print(f"[warn] file mancante: {p}")
                    else:
                        print(f"[warn] policy_net[{i}] non esiste nell'istanza: salto.")

                # target policy
                tp = tmp / f"target_policy_net_{i}.pt"
                if hasattr(self, "target_policy_net") and len(self.target_policy_net) > i and tp.exists():
                    _load_into(self.target_policy_net[i], tp)
                else:
                    if not tp.exists():
                        print(f"[warn] file mancante: {tp}")
                    else:
                        print(f"[warn] target_policy_net[{i}] non esiste nell'istanza: salto.")

            # 5) critic e target_critic (globali)
            c = tmp / "critic.pt"
            if hasattr(self, "critic") and c.exists():
                _load_into(self.critic, c)
            else:
                if not c.exists():
                    print("[warn] critic.pt non trovato.")
                else:
                    print("[warn] self.critic non presente: salto.")

            tc = tmp / "target_critic.pt"
            if hasattr(self, "target_critic") and tc.exists():
                _load_into(self.target_critic, tc)
            else:
                if not tc.exists():
                    print("[warn] target_critic.pt non trovato.")
                else:
                    print("[warn] self.target_critic non presente: salto.")

        print(".... AAC-MADRL models loaded ✓")


    def predict(self, observations: List[List[float]], deterministic: bool = None):
        r"""Provide actions for current time step.

        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` >= :attr:`time_step`
        else will use policy to sample actions.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[float]
            Action values
        """

        deterministic = False if deterministic is None else deterministic

        if self.time_step > self.end_exploration_time_step or deterministic:
            actions = self.get_post_exploration_prediction(observations, deterministic)

        else:
            actions = self.get_exploration_prediction(observations)
        self.actions = actions
        self.next_time_step()

        return actions

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float],
               next_observations: List[List[float]], terminated: bool, truncated: bool):
        r"""Update replay buffer.

        Parameters
        ----------
        observations : List[List[float]]
            Previous time step observations.
        actions : List[List[float]]
            Previous time step actions.
        reward : List[float]
            Current time step reward.
        next_observations : List[List[float]]
            Current time step observations.
        done : bool
            Indication that episode has ended.
        """

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).

        self.update_buffer(observations, actions, reward, next_observations, terminated)
        for _ in range(self.update_per_time_step):
            full_buffer_agents = 0
            all_obs, all_acs, all_rews, all_next_obs, all_done = [], [], [], [], []
            for agent in range(len(self.action_dimension)):
                if self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[agent]):
                    o, a, r, n, d = self.replay_buffer[agent].sample(self.batch_size)
                    tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                    #obs = tensor(o).to(self.device)
                    obs = torch.tensor(o, device=self.device)
                    #next_obs = tensor(n).to(self.device)
                    next_obs = torch.tensor(n, device=self.device)
                    a_discrete = self.discretize_actions(a, agent=agent)
                    actions_list = [torch.tensor(np.array([item[i] for item in a_discrete]), dtype=torch.float32).to(self.device) for i in range(self.action_dimension[agent])]
                    #rews = tensor(r).unsqueeze(1).to(self.device)
                    rews = torch.tensor(r, device=self.device).unsqueeze(1)
                    #done = tensor(d).unsqueeze(1).to(self.device)
                    terminated = torch.tensor(d, device=self.device).unsqueeze(1)
                    all_obs.append(obs)
                    all_acs.append(actions_list)
                    all_rews.append(rews)
                    all_next_obs.append(next_obs)
                    all_done.append(terminated)
                    full_buffer_agents += 1

            if full_buffer_agents == self.nagents:
                sample = (all_obs, all_acs, all_rews, all_next_obs, all_done)
                self.update_critic(sample)
                self.update_policies(sample)
                self.update_all_targets()

    def get_post_exploration_prediction(self, observations: List[List[float]], deterministic: bool) -> List[
        List[float]]:
        """Action sampling using policy, post-exploration time step"""

        actions = []
        
        for i, o in enumerate(observations):
            o = self.get_encoded_observations(i, o)
            o = self.get_normalized_observations(i, o)
            o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
            result = self.policy_net[i](o, self.action_dim[i], sample=not deterministic)
            actions.append(result)
        # MAKING  DISCRETE TO CONTINUOUS
        acs = self.make_continuous(actions, self.defined_actions)
        return acs

    def get_exploration_prediction(self, observations: List[List[float]]) -> List[List[float]]:
        """Return randomly sampled actions from `action_space` multiplied by :attr:`action_scaling_coefficient`."""
        sampled_actions = [list(self.action_scaling_coefficient * s.sample()) for s in self.action_space]
        effective_actions = self.make_continuous(self.discretize_actions(sampled_actions), self.defined_actions)
        return effective_actions

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index] * np.array(observations, dtype=float)) if j != None],
                        dtype=float)

    def get_normalized_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        try:
            return (np.array(observations, dtype=float) - self.norm_mean[index]) / self.norm_std[index]
        except:
            # self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[i])
            print('obs:', observations)
            print('mean:', self.norm_mean[index])
            print('std:', self.norm_std[index])
            print(self.time_step, self.standardize_start_time_step, self.batch_size, len(self.replay_buffer[0]))
            assert False

    def get_normalized_reward(self, index: int, reward: float) -> float:
        return (reward - self.r_norm_mean[index]) / self.r_norm_std[index]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        obs = [ob.to(torch.float32).to(self.device) for ob in obs]
        next_obs = [ob.to(torch.float32).to(self.device) for ob in next_obs]
        # Q loss
        next_acs = []
        next_log_pis = []
        action_dims = [self.action_dim[a_i] for a_i in range(self.nagents)]
        for a_i, pi, ob, action_dim in zip(range(self.nagents), self.target_policy_net, next_obs, action_dims):
            curr_next_ac, curr_next_log_pi = pi(ob, action_dim, return_log_pi=True, sample=True) ####change sample
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            '''
            target_q = (rews[a_i].view(-1, 1) +
                        self.discount * nq *
                        (1 - dones[a_i].view(-1, 1)))
            '''
            target_q = (rews[a_i].view(-1, 1).float() +
                        self.discount * nq *
                        (~dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi/ self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        obs = [ob.to(torch.float32).to(self.device) for ob in obs]
        samp_acs, all_probs, all_log_pis, all_pol_regs = [], [], [], []
        action_dims = [self.action_dim[a_i] for a_i in range(self.nagents)]

        for a_i, pi, ob, action_dim in zip(range(self.nagents), self.policy_net, obs, action_dims):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, action_dim, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True, sample=True) ####change sample
            # logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
            # self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            probs_tensor = torch.cat(probs, dim=1)
            v = (all_q * probs_tensor).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_net[a_i].parameters(), 0.5)
            self.policy_optimizer[a_i].step()
            self.policy_optimizer[a_i].zero_grad()

            '''
            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)
            '''

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in range(self.nagents):
            soft_update(self.target_policy_net[a], self.policy_net[a], self.tau)

    def discretize_actions(self, acs, agent: Optional[int] = None):
        """
        Discretize actions
        """
        if agent is not None:
            keys = self.action_names[agent]
            bins_agent = [
                (np.append(np.arange(0, 1, 1/self.classes[k]), 1)
                if k in ('cooling_device', 'heating_device')
                else np.append(np.arange(-1, 1, 2/self.classes[k]), 1))
                for k in keys
            ]

            arr = np.atleast_2d(acs)  # shape: (batch, n_azioni)
            digitized = [
                [np.digitize(x, bins_agent[j], right=True) for j, x in enumerate(row)]
                for row in arr
            ]
            one_hot = [
                [np.eye(self.classes[k])[int(idx)-1] for idx, k in zip(row, keys)]
                for row in digitized
            ]
            return one_hot
        bins = [
        [
            (np.append(np.arange(0, 1, 1 / self.classes[key]), 1)
             if key in ['cooling_device', 'heating_device']
             else np.append(np.arange(-1, 1, 2 / self.classes[key]), 1))
            for key in self.action_names[agent]
        ]
        for agent in range(self.nagents)
        ]
        digitized = [
            [np.digitize(ac, bins_agent[j], right=True) for j, ac in enumerate(agent_acs)]
            for bins_agent, agent_acs in zip(bins, acs)
        ]

        num_classes_per_agent = [
        [self.classes[key] for key in self.action_names[agent]]
        for agent in range(self.nagents)
    ]
        one_hot_list = [[np.eye(num_classes_per_agent[agent][j])[(digitized_ac-1).astype(int)] for j, digitized_ac in enumerate(agent_digitized)] 
                    for agent, agent_digitized in enumerate(digitized)]

        return one_hot_list

    def make_continuous(self, acs, defined_actions):
        """
        Convert one-hot encoded actions back to their original form
        """
        continuous_actions = []
        for agent, agent_acs in enumerate(acs):
            agent_vals = []
            for j, ac in enumerate(agent_acs):
                key = self.action_names[agent][j]
                idx = np.argmax(ac.cpu().numpy() if torch.is_tensor(ac) else ac)
                agent_vals.append(defined_actions[key][idx])
            continuous_actions.append(agent_vals)
        #continuous_actions = [[defined_actions[j][np.argmax(ac.cpu().numpy() if torch.is_tensor(ac) else ac)] 
                               #for j, ac in enumerate(agent_acs)] for agent_acs in acs]
        return continuous_actions

    def update_buffer(self, observations, actions, reward, next_observations, done):
        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o = self.get_encoded_observations(i, o)
            n = self.get_encoded_observations(i, n)

            if self.normalized[i]:
                o = self.get_normalized_observations(i, o)
                n = self.get_normalized_observations(i, n)
                r = self.get_normalized_reward(i, r)
            else:
                pass

            self.replay_buffer[i].push(o, a, r, n, done)

            if self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[i]):
                if not self.normalized[i]:
                    # calculate normalized observations and rewards
                    X = np.array([j[0] for j in self.replay_buffer[i].buffer], dtype=float)
                    self.norm_mean[i] = np.nanmean(X, axis=0)
                    self.norm_std[i] = np.nanstd(X, axis=0) + 1e-5
                    R = np.array([j[2] for j in self.replay_buffer[i].buffer], dtype=float)
                    self.r_norm_mean[i] = np.nanmean(R, dtype=float)
                    self.r_norm_std[i] = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5

                    # update buffer with normalization
                    self.replay_buffer[i].buffer = [(
                        np.hstack(self.get_normalized_observations(i, o).reshape(1, -1)[0]),
                        a,
                        self.get_normalized_reward(i, r),
                        np.hstack(self.get_normalized_observations(i, n).reshape(1, -1)[0]),
                        d
                    ) for o, a, r, n, d in self.replay_buffer[i].buffer]
                    self.normalized[i] = True

                else:
                    pass

class AAC_MADRL_RBC(AAC_MADRL):
    r"""Uses :py:class:`citylearn.agents.rbc.RBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    rbc: RBC
        :py:class:`citylearn.agents.rbc.RBC` or child class, used to select actions during exploration.
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, rbc: RBC = None, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.__set_rbc(rbc, **kwargs)

    @property
    def rbc(self) -> RBC:
        """:py:class:`citylearn.agents.rbc.RBC` or child class, used to select actions during exploration."""

        return self.__rbc
    
    def __set_rbc(self, rbc: RBC, **kwargs):
        if rbc is None:
            rbc = RBC(self.env, **kwargs)
        
        elif isinstance(rbc, RBC):
            pass

        else:
            rbc = rbc(self.env, **kwargs)
        
        self.__rbc = rbc

    def get_exploration_prediction(self, observations: List[float]) -> List[float]:
        """Return actions using :class:`RBC`."""

        return self.rbc.predict(observations)


