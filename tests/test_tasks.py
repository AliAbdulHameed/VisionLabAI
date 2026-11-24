from visionlab.tasks import TASKS


def test_task_keys_match():
    assert all(key == task.key for key, task in TASKS.items())


def test_model_tasks_have_weights():
    assert TASKS["detect"].weights == "yolo11n.pt"
    assert TASKS["pose"].weights == "yolo26n-pose.pt"
    assert not TASKS["sketch"].uses_model


def test_every_task_has_icon_and_tips():
    for task in TASKS.values():
        assert task.icon and task.tips
        assert task.label == f"{task.name} · {task.model}"
