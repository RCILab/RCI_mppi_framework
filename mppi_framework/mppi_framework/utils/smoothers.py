import torch
import torch.nn.functional as F

class Smoothers:
    @staticmethod
    def moving_average_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        Moving Average Filter for trajectory smoothing.
        Args:
            x: Input tensor of shape (B, T, U) or (T, U)
            kernel_size: Window size (must be odd, e.g., 3, 5)
        Returns:
            Smoothed tensor with same shape as input (padded to maintain size)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # 홀수 강제

        orig_shape = x.shape
        if x.dim() == 2: # (T, U) -> (1, T, U)로 변환
            x = x.unsqueeze(0)
        
        B, T, U = x.shape
        padding = kernel_size // 2
        
        # (B, T, U) -> (B, U, T)로 변경 (Conv1d는 마지막 차원을 시간으로 보지 않음)
        x_perm = x.permute(0, 2, 1) 
        
        # 가장자리 패딩 (ReplicationPad가 튀는 값 방지에 유리)
        # pad args: (left, right)
        x_pad = F.pad(x_perm, (padding, padding), mode='replicate')
        
        # 커널 생성 (모든 가중치가 1/N)
        weight = torch.ones(U, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
        
        # Group Conv1d 적용 (각 채널(U)별로 독립적으로 필터링)
        # groups=U 로 설정하면 depthwise convolution이 됨
        out = F.conv1d(x_pad, weight, groups=U)
        
        # 다시 원래 shape (B, T, U)로 복구
        return out.permute(0, 2, 1).view(orig_shape)

    @staticmethod
    def savitzky_golay_filter(x: torch.Tensor, window_length: int = 5, polyorder: int = 2) -> torch.Tensor:
        """
        Savitzky-Golay Filter for trajectory smoothing.
        Preserves peak values better than Moving Average.
        Args:
            x: Input tensor of shape (B, T, U) or (T, U)
            window_length: Window size (must be odd)
            polyorder: Order of the polynomial to fit (typically 2 or 3)
        Returns:
            Smoothed tensor with same shape as input
        """
        if window_length % 2 == 0:
            window_length += 1
            
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        B, T, U = x.shape
        
        # 계수 계산 (미리 계산된 계수를 쓰면 빠르지만, 범용성을 위해 여기서 계산)
        # SG 필터 계수는 (window_length,) 크기의 벡터임
        half_window = (window_length - 1) // 2
        
        # 1. Vandermonde Matrix 생성
        k = torch.arange(-half_window, half_window + 1, dtype=x.dtype, device=x.device)
        M = k.unsqueeze(1) ** torch.arange(polyorder + 1, dtype=x.dtype, device=x.device).unsqueeze(0) # [window, poly+1]
        
        # 2. Least Squares 해 (M_pinv)의 0번째 행이 스무딩 계수임
        # coeff = (M^T M)^-1 M^T * y -> 0번째 미분(값 자체)에 대한 계수만 필요
        m_pinv = torch.pinverse(M) # [poly+1, window]
        coeffs = m_pinv[0]         # [window] -> 이것이 Convolution 커널이 됨
        
        # 3. Conv1d 적용 준비
        x_perm = x.permute(0, 2, 1) # (B, U, T)
        padding = half_window
        x_pad = F.pad(x_perm, (padding, padding), mode='replicate')
        
        # 커널 shape: (out_channels, in_channels/groups, kernel_size)
        # 각 채널마다 동일한 coeffs를 적용
        weight = coeffs.view(1, 1, -1).repeat(U, 1, 1) # [U, 1, window]
        
        out = F.conv1d(x_pad, weight, groups=U)
        
        return out.permute(0, 2, 1).view(orig_shape)