from pathlib import Path

from visionlab.tasks import TASKS


def test_task_keys_match():
    assert all(key == task.key for key, task in TASKS.items())


def test_model_tasks_point_at_bundled_models():
    for key in ("detect", "pose"):
        model_dir = Path(TASKS[key].model_dir)
        assert list(model_dir.glob("*.xml")) and list(model_dir.glob("*.bin"))
    assert not TASKS["sketch"].uses_model


def test_every_task_has_icon_and_tips():
    for task in TASKS.values():
        assert task.icon and task.tips
        assert task.label == f"{task.name} · {task.model}"
