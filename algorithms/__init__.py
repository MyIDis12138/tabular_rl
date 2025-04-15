# algorithms/__init__.py
from .q_learning import QLearningAgent
from .sarsa import SarsaAgent
from .rmax import RMaxAgent
from .actor_critic import ActorCriticAgent

# You might want a mapping for easier loading in run.py
AGENT_MAP = {
    'qlearning': QLearningAgent,
    'sarsa': SarsaAgent,
    'rmax': RMaxAgent,
    'actorcritic': ActorCriticAgent,
}
