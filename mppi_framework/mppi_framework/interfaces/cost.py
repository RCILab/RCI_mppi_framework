from abc import ABC, abstractmethod

class BaseCost(ABC):
    @abstractmethod
    def stage(self, X, U):
        """
        X: [B,T,S], U: [B,T,U]  -> returns C_stage: [B,T]
        """
        ...

    def terminal(self, x_T):
        """terminal cost (optional): default 0"""
        return x_T.new_zeros(x_T.shape[0])
