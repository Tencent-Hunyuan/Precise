import os


def _parse_visible_devices():
    raw = os.environ.get("PRECISE_SDE_GENEVAL_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


VISIBLE_DEVICES = _parse_visible_devices()
NUM_DEVICES = int(os.environ.get("PRECISE_SDE_GENEVAL_NUM_DEVICES", str(len(VISIBLE_DEVICES) or 1)))
USED_DEVICES = set()
HOST = os.environ.get("PRECISE_SDE_GENEVAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("PRECISE_SDE_GENEVAL_PORT", "18085"))


def pre_fork(server, worker):
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    if VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[worker.device_id]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)


def child_exit(server, worker):
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)


bind = f"{HOST}:{PORT}"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120
