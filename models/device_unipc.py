"""FlowUniPCMultistepScheduler.step on device tensors, without a host round trip.

The scheduler's tensor math is linear combinations of the sample and the stored model outputs
with scalar coefficients that depend only on the sigma schedule. The coefficients stay on the CPU;
only the combinations run on the device. The two places where the upstream code builds small
tensors on the sample's device and solves a linear system there (`rks`, `b`, `rhos`,
`torch.linalg.solve`) run on the CPU instead, and the einsum over the history is written out as a
sum of scalar x tensor terms.

Same arithmetic and operation order as upstream (solver_order 2, predict_x0, flow_prediction,
bh1/bh2), including how PyTorch applies upstream's 0-dim fp32 coefficients to bf16 tensors (see
`_c`). On the CPU this reproduces upstream bit for bit (scripts/validate/unipc_device_check.py); on
the device it differs only by the device's elementwise rounding.
"""
import torch

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def _f(x):
    return float(x) if isinstance(x, torch.Tensor) else x


class DeviceFlowUniPC(FlowUniPCMultistepScheduler):

    def _c(self, x, dtype):
        """A 0-dim fp32 coefficient as upstream applies it from the LEFT of a multiply with a
        `dtype` tensor: type promotion casts it to `dtype` first (PyTorch keeps the original
        precision only for a scalar on the right, as in the D1 division, which uses float(rk))."""
        return float(torch.tensor(_f(x), dtype=torch.float32).to(dtype))

    def _coeffs(self, h, rks):
        """R matrix, b vector, h_phi_1 and B_h for UniPC-bh (CPU math, as upstream)."""
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
        R, b = [], []
        rks_t = torch.tensor(rks)
        for i in range(1, len(rks) + 1):
            R.append(torch.pow(rks_t, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        return torch.stack(R), torch.tensor(b), h_phi_1, B_h

    def _lambda(self, sigma):
        alpha, sigma = self._sigma_to_alpha_sigma_t(sigma)
        return torch.log(alpha) - torch.log(sigma), alpha, sigma

    def multistep_uni_p_bh_update(self, model_output, *args, sample=None, order=None, **kwargs):
        assert self.solver_p is None
        m = self.model_outputs
        m0, x = m[-1], sample
        c = lambda v: self._c(v, x.dtype)  # noqa: E731
        lambda_t, alpha_t, sigma_t = self._lambda(self.sigmas[self.step_index + 1])
        lambda_s0, alpha_s0, sigma_s0 = self._lambda(self.sigmas[self.step_index])
        h = lambda_t - lambda_s0
        rks, d1 = [], []
        for i in range(1, order):
            lambda_si, _, _ = self._lambda(self.sigmas[self.step_index - i])
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1.append((m[-(i + 1)] - m0) / _f(rk))
        rks.append(1.0)
        R, b, h_phi_1, B_h = self._coeffs(h, rks)
        pred_res = None
        if d1:
            if order == 2:
                rhos_p = [0.5]
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(x.dtype).float().tolist()
            for r, d in zip(rhos_p, d1):
                pred_res = r * d if pred_res is None else pred_res + r * d
        if self.predict_x0:
            x_t_ = c(sigma_t / sigma_s0) * x - c(alpha_t * h_phi_1) * m0
            x_t = x_t_ - c(alpha_t * B_h) * pred_res if pred_res is not None else x_t_
        else:
            x_t_ = c(alpha_t / alpha_s0) * x - c(sigma_t * h_phi_1) * m0
            x_t = x_t_ - c(sigma_t * B_h) * pred_res if pred_res is not None else x_t_
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output, *args, last_sample=None, this_sample=None,
                                  order=None, **kwargs):
        m = self.model_outputs
        m0, x, model_t = m[-1], last_sample, this_model_output
        c = lambda v: self._c(v, x.dtype)  # noqa: E731
        lambda_t, alpha_t, sigma_t = self._lambda(self.sigmas[self.step_index])
        lambda_s0, alpha_s0, sigma_s0 = self._lambda(self.sigmas[self.step_index - 1])
        h = lambda_t - lambda_s0
        rks, d1 = [], []
        for i in range(1, order):
            lambda_si, _, _ = self._lambda(self.sigmas[self.step_index - (i + 1)])
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1.append((m[-(i + 1)] - m0) / _f(rk))
        rks.append(1.0)
        R, b, h_phi_1, B_h = self._coeffs(h, rks)
        rhos_c = [0.5] if order == 1 else torch.linalg.solve(R, b).to(x.dtype).float().tolist()
        corr_res = None
        for r, d in zip(rhos_c[:-1], d1):
            corr_res = r * d if corr_res is None else corr_res + r * d
        d1_t = model_t - m0
        res = rhos_c[-1] * d1_t if corr_res is None else corr_res + rhos_c[-1] * d1_t
        if self.predict_x0:
            x_t_ = c(sigma_t / sigma_s0) * x - c(alpha_t * h_phi_1) * m0
            x_t = x_t_ - c(alpha_t * B_h) * res
        else:
            x_t_ = c(alpha_t / alpha_s0) * x - c(sigma_t * h_phi_1) * m0
            x_t = x_t_ - c(sigma_t * B_h) * res
        return x_t.to(x.dtype)

    def convert_model_output(self, model_output, *args, sample=None, **kwargs):
        assert self.predict_x0 and self.config.prediction_type == "flow_prediction" \
            and not self.config.thresholding
        return sample - self._c(self.sigmas[self.step_index], sample.dtype) * model_output
