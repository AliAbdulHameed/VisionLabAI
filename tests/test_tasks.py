from visionlab.tasks import TASKS


def test_task_keys_match():
    assert all(key == task.key for key, task in TASKS.items())


def test_model_tasks_have_weights():
    assert TASKS["detect"].weights == "yolo11n.pt"
    assert TASKS["pose"].weights == "yolo26n-pose.pt"
    assert not TASKS["sketch"].uses_model
