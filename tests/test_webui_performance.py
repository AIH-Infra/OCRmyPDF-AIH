from ocrmypdf.aih_webui_performance import (
    _detect_nvidia_gpus,
    recommend_performance,
)


def _hardware(*, cores=8, available_gib=16, is_32bit=False, gpus=None):
    return {
        "cpu_logical_cores": cores,
        "cpu_physical_cores": None,
        "memory_total_bytes": available_gib * 1024**3,
        "memory_available_bytes": available_gib * 1024**3,
        "is_32bit": is_32bit,
        "gpus": gpus,
    }


def test_tesseract_concurrency_respects_cpu_and_memory_budget_and_file_count():
    facts = _hardware(cores=8, available_gib=9)
    one_file = recommend_performance("Tesseract", facts, file_count=1)
    several_files = recommend_performance("Tesseract", facts, file_count=3)

    assert one_file["file_concurrency"] == 1
    assert several_files["file_concurrency"] == 3
    assert several_files["jobs"] >= 1
    assert several_files["jobs"] * several_files["file_concurrency"] <= 6
    assert one_file["jobs"] * one_file["file_concurrency"] <= 6


def test_screenai_defaults_to_page_parallel_for_any_queue_size():
    facts = _hardware(cores=24, available_gib=48)
    for count in (1, 3, 50):
        for level in ("low", "medium", "high"):
            plan = recommend_performance("Chrome ScreenAI（本地）", facts, file_count=count, level=level)
            assert plan["chrome_concurrency_mode"] == "page_parallel"
            assert plan["chrome_page_workers"] == plan["jobs"] > 1
            assert plan["file_concurrency"] == 1


def test_screenai_file_pool_requires_explicit_selection():
    facts = _hardware(cores=24, available_gib=48)
    plan = recommend_performance("Chrome ScreenAI", facts, file_count=50,
                                 screenai_concurrency="file_parallel")
    assert plan["chrome_concurrency_mode"] == "file_parallel"
    assert plan["jobs"] == 1
    assert plan["chrome_file_concurrency"] == plan["file_concurrency"] > 1


def test_low_memory_and_32_bit_hosts_stay_conservative():
    for facts in (
        _hardware(cores=16, available_gib=1),
        _hardware(cores=16, available_gib=16, is_32bit=True),
    ):
        result = recommend_performance("Tesseract", facts, file_count=10)
        assert result["jobs"] == 1
        assert result["file_concurrency"] == 1


def test_paddle_gpu_requires_a_resolved_gpu_device():
    gpu = [{"name": "NVIDIA Test", "vram_bytes": 8 * 1024**3, "cuda_supported": True}]
    facts = _hardware(cores=12, available_gib=24, gpus=gpu)

    automatic = recommend_performance("PaddleOCR（本地全链路）", facts, device="auto")
    selected = recommend_performance("PaddleOCR（本地全链路）", facts, device="gpu:0")

    assert automatic["jobs"] > 1
    assert "按 CPU 设备规划" in automatic["reason"]
    assert selected["jobs"] == selected["file_concurrency"] == 1
    assert selected["paddle_ocr_cpu_threads"] >= 1
    assert selected["paddle_ocr_rec_batch_size"] == 16


def test_unknown_hardware_is_explicit_and_recommends_one_worker():
    result = recommend_performance("Tesseract", {}, file_count=4)

    assert result["jobs"] == result["file_concurrency"] == 1
    assert "CPU 逻辑核数未知" in result["reason"]
    assert "可用内存读数未知" in result["reason"]


def test_nvidia_smi_metadata_is_parsed_without_gpu_execution(monkeypatch):
    calls = []
    monkeypatch.setattr("ocrmypdf.aih_webui_performance.shutil.which", lambda _: "nvidia-smi")

    def run_query(*args):
        calls.append(args)
        if "--query-gpu=name,memory.total,driver_version" in args:
            return "NVIDIA Example, 8192, 555.42\n"
        return "| NVIDIA-SMI 555.42 | CUDA Version: 12.5 |\n"

    monkeypatch.setattr("ocrmypdf.aih_webui_performance._run_system_query", run_query)
    gpus = _detect_nvidia_gpus()

    assert gpus == [
        {
            "name": "NVIDIA Example",
            "vram_bytes": 8192 * 1024**2,
            "driver_version": "555.42",
            "cuda_supported": True,
            "cuda_version": "12.5",
        }
    ]
    assert len(calls) == 2


def test_resource_tiers_are_ordered_for_each_backend():
    facts = _hardware(cores=24, available_gib=48)
    for backend in ("Tesseract", "Chrome ScreenAI", "PaddleOCR", "Calamari"):
        plans = [recommend_performance(backend, facts, level=level, file_count=12)
                 for level in ("low", "medium", "high")]
        totals = [p["jobs"] * p["file_concurrency"] for p in plans]
        assert totals == sorted(totals)
        assert totals[-1] < facts["cpu_logical_cores"]
        if backend == "Calamari":
            assert [p["preflight_chunk"] for p in plans] == [5, 10, 20]
            assert plans[0]["preflight_workers"] < plans[-1]["preflight_workers"]
        elif backend == "PaddleOCR":
            assert [p["paddle_ocr_rec_batch_size"] for p in plans] == [2, 4, 8]
            for p in plans:
                assert p["jobs"] * p["paddle_ocr_cpu_threads"] <= facts["cpu_logical_cores"]
        else:
            assert totals[0] < totals[-1]


def test_tesseract_combined_engine_and_languages_reduce_memory_concurrency():
    facts = _hardware(cores=32, available_gib=24)
    single = recommend_performance("Tesseract", facts, tesseract_oem=1, language_count=1)
    combined = recommend_performance("Tesseract", facts, tesseract_oem=2, language_count=3)
    assert combined["jobs"] < single["jobs"]
    assert "OEM 2、3 种语言" in combined["reason"]


def test_paddle_gpu_tiers_limit_batch_by_selected_device_memory():
    facts = _hardware(cores=24, available_gib=48, gpus=[
        {"vram_bytes": 4 * 1024**3}, {"vram_bytes": 24 * 1024**3}])
    plans = [recommend_performance("PaddleOCR", facts, level=level, device="gpu:1")
             for level in ("low", "medium", "high")]
    assert [p["paddle_ocr_rec_batch_size"] for p in plans] == [4, 16, 32]
    assert all(p["jobs"] == p["file_concurrency"] == 1 for p in plans)
    small = recommend_performance("PaddleOCR", facts, level="high", device="gpu:0")
    assert small["paddle_ocr_rec_batch_size"] == 4


def test_screenai_light_model_uses_smaller_memory_budget():
    facts = _hardware(cores=32, available_gib=12)
    standard = recommend_performance("Chrome ScreenAI", facts)
    light = recommend_performance("Chrome ScreenAI", facts, screenai_light=True)
    assert light["jobs"] > standard["jobs"]


def test_system_probe_decodes_utf8_independently_of_windows_locale(monkeypatch):
    import sys
    from ocrmypdf import aih_webui_performance as perf
    run = perf.subprocess.run
    def gbk_default(*args, **kwargs):
        kwargs.setdefault("encoding", "gbk")
        return run(*args, **kwargs)
    monkeypatch.setattr(perf.subprocess, "run", gbk_default)
    result = perf._run_system_query(sys.executable, "-c",
        "import sys;sys.stdout.buffer.write('GPU \u2713'.encode('utf-8'))")
    assert result == "GPU \u2713"
