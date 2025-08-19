import threading

_thread_local = threading.local()


def set_cancellation_flag(node_name):
    _thread_local.__setattr__('cancel_requested', True)


def is_cancelled() -> bool:
    return getattr(_thread_local, 'cancel_requested', False)
