# algorithms/__init__.py
from .base_agent import BaseAgent
from .q_learning import QLearningAgent
from .sarsa import SarsaAgent
from .rmax import RMaxAgent
from .actor_critic import ActorCriticAgent

# Mapping for easier loading in run.py
AGENT_MAP = {
    'base': BaseAgent,
    'qlearning': QLearningAgent,
    'sarsa': SarsaAgent,
    'rmax': RMaxAgent,
    'actorcritic': ActorCriticAgent,
}