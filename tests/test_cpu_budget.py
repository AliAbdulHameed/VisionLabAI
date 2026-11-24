from visionlab.inference import cgroup_cpu_quota, inference_threads


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_quota(tmp_path):
    write(tmp_path / "cpu.max", "150000 100000\n")
    assert cgroup_cpu_quota(str(tmp_path), str(tmp_path / "none")) == 1.5


def test_cgroup_v2_unlimited(tmp_path):
    write(tmp_path / "cpu.max", "max 100000\n")
    assert cgroup_cpu_quota(str(tmp_path), str(tmp_path / "none")) is None


def test_cgroup_v1_quota_in_nested_group(tmp_path):
    proc = tmp_path / "proc_cgroup"
    write(proc, "4:memory:/x\n1:cpu,cpuacct:/app\n")
    write(tmp_path / "cpu,cpuacct" / "app" / "cpu.cfs_quota_us", "200000")
    write(tmp_path / "cpu,cpuacct" / "app" / "cpu.cfs_period_us", "100000")
    assert cgroup_cpu_quota(str(tmp_path), str(proc)) == 2.0


def test_cgroup_v1_unlimited(tmp_path):
    proc = tmp_path / "proc_cgroup"
    write(proc, "1:cpu:/\n")
    write(tmp_path / "cpu" / "cpu.cfs_quota_us", "-1")
    write(tmp_path / "cpu" / "cpu.cfs_period_us", "100000")
    assert cgroup_cpu_quota(str(tmp_path), str(proc)) is None


def test_no_cgroup_info(tmp_path):
    assert cgroup_cpu_quota(str(tmp_path), str(tmp_path / "missing")) is None


def test_inference_threads_leave_a_core_for_video():
    assert inference_threads(1) == 1
    assert inference_threads(2) == 1
    assert inference_threads(4) == 3
