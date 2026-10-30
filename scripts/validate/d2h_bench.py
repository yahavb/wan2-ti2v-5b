"""How do device->host copies synchronize on torch-neuronx (one core)? For the VAE chunk ship:
does .cpu() wait for everything queued, does to('cpu', non_blocking=True) return early, do
torch.neuron.Event record/synchronize work, and does a .cpu() in a helper thread block the main
thread's queuing? A queue of ~WORK_MS of matmuls is in flight in every test.
NEURON_RT_VISIBLE_CORES=<one core> python3 d2h_bench.py
"""
import threading
import time

import torch

dev = torch.device("neuron")
torch.set_grad_enabled(False)
a = torch.randn(4096, 4096).to(torch.bfloat16).to(dev)
small = torch.randn(3, 16, 88, 544).to(torch.bfloat16).to(dev)  # one rank's chunk (4.6 MB)
mm = torch.compile(lambda t: (t @ t) * 1e-3, backend="neuron", dynamic=False)


def queue_work(n):
    t = a
    for _ in range(n):
        t = mm(t)
    return t


queue_work(2)
torch.neuron.synchronize()
t0 = time.perf_counter()
queue_work(40)
tq = time.perf_counter() - t0
torch.neuron.synchronize()
tw = time.perf_counter() - t0
print(f"[d2h] queue 40 matmuls: host {tq * 1e3:.1f} ms, device done after {tw * 1e3:.1f} ms", flush=True)
N = 40

# 1. .cpu() behind queued work
queue_work(N)
t0 = time.perf_counter()
y = small.cpu()
print(f"[d2h] .cpu() with {N} matmuls queued: {1e3 * (time.perf_counter() - t0):.1f} ms", flush=True)
torch.neuron.synchronize()

# 2. non_blocking
queue_work(N)
t0 = time.perf_counter()
try:
    y2 = small.to("cpu", non_blocking=True)
    t1 = time.perf_counter()
    ok = torch.equal(y2, y)
    print(f"[d2h] to(cpu, non_blocking): returned after {1e3 * (t1 - t0):.1f} ms, equal-now={ok}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[d2h] non_blocking failed: {type(e).__name__}: {e}", flush=True)
torch.neuron.synchronize()

# 3. event recorded after the queue
queue_work(N)
t0 = time.perf_counter()
try:
    ev = torch.neuron.Event()
    ev.record()
    t1 = time.perf_counter()
    q = ev.query()
    ev.synchronize()
    t2 = time.perf_counter()
    print(f"[d2h] Event.record {1e3 * (t1 - t0):.1f} ms, query={q}, synchronize done at {1e3 * (t2 - t0):.1f} ms",
          flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[d2h] Event failed: {type(e).__name__}: {e}", flush=True)
torch.neuron.synchronize()

# 4. .cpu() in a helper thread while the main thread keeps queuing
res = {}


def helper():
    t0 = time.perf_counter()
    res["y"] = small.cpu()
    res["t"] = time.perf_counter() - t0


queue_work(10)
th = threading.Thread(target=helper)
t0 = time.perf_counter()
th.start()
queue_work(N)
tq = time.perf_counter() - t0
th.join()
tj = time.perf_counter() - t0
torch.neuron.synchronize()
tall = time.perf_counter() - t0
print(f"[d2h] helper .cpu() behind 10 queued, main queues {N} more: main queued in {tq * 1e3:.1f} ms, "
      f"helper copy took {res['t'] * 1e3:.1f} ms (joined at {tj * 1e3:.1f}), all done {tall * 1e3:.1f} ms",
      flush=True)

# 5. event recorded behind 10, helper waits on the event then copies, main keeps queuing
queue_work(10)
try:
    ev = torch.neuron.Event()
    ev.record()

    def helper2():
        t0 = time.perf_counter()
        ev.synchronize()
        res["te"] = time.perf_counter() - t0
        res["y"] = small.cpu()
        res["t"] = time.perf_counter() - t0
    th = threading.Thread(target=helper2)
    t0 = time.perf_counter()
    th.start()
    queue_work(N)
    tq = time.perf_counter() - t0
    th.join()
    tj = time.perf_counter() - t0
    torch.neuron.synchronize()
    print(f"[d2h] helper waits on event (behind 10) then .cpu(), main queues {N}: main {tq * 1e3:.1f} ms, "
          f"event at {res['te'] * 1e3:.1f}, copy done at {res['t'] * 1e3:.1f} ms (joined {tj * 1e3:.1f})", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[d2h] event+helper failed: {type(e).__name__}: {e}", flush=True)

# 6. copy on a side stream after the event
queue_work(10)
try:
    ev = torch.neuron.Event()
    ev.record()
    s = torch.neuron.Stream()
    t0 = time.perf_counter()
    with torch.neuron.stream(s):
        s.wait_event(ev)
        y3 = small.to("cpu", non_blocking=True)
    t1 = time.perf_counter()
    queue_work(N)
    s.synchronize()
    t2 = time.perf_counter()
    print(f"[d2h] side-stream copy: issued in {1e3 * (t1 - t0):.1f} ms, side stream done at {1e3 * (t2 - t0):.1f} ms, "
          f"equal={torch.equal(y3, y)}", flush=True)
    torch.neuron.synchronize()
except Exception as e:  # noqa: BLE001
    print(f"[d2h] side stream failed: {type(e).__name__}: {e}", flush=True)
