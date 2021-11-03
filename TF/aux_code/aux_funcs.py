import os
import threading


def launch_tb(log_dir):
    os.system(f'tensorboard --logdir={log_dir}')

    th = threading.Thread(
        target=lambda logdir: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    th.start()
