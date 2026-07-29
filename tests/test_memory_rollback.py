import json
from pathlib import Path

import pytest

import game_window_ocr.commentary as commentary_module
from game_window_ocr import memory_store
from game_window_ocr import memory_rollback
from game_window_ocr.commentary import TextResult


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _overall(session_count: int) -> dict:
    return {
        "schema_version": 1,
        "game_title": "かまいたちの夜",
        "updated_at": f"2026-07-2{session_count}T20:00:00+09:00",
        "session_count": session_count,
        "story_summary": f"{session_count}回目までのあらすじ",
        "characters": [],
        "important_choices": [],
        "unresolved_threads": [],
        "current_state": "ペンションで待機中",
        "commentator_perspective": "犯人はまだ不明",
        "next_start_point": "次の選択肢から",
        "last_session_summary": f"{session_count}回目の実況",
    }


class TestBackup:
    def test_直前の全体記憶をoverall_previousへ退避する(self, tmp_path) -> None:
        overall_path = tmp_path / "overall.json"
        _write(overall_path, _overall(1))

        backup_path = memory_store.backup_overall_memory(overall_path)

        assert backup_path == tmp_path / memory_store.PREVIOUS_FILENAME
        assert json.loads(backup_path.read_text(encoding="utf-8")) == _overall(1)
        assert json.loads(overall_path.read_text(encoding="utf-8")) == _overall(1)

    def test_保持するのは1回前だけで古い退避は上書きされる(self, tmp_path) -> None:
        overall_path = tmp_path / "overall.json"
        _write(overall_path, _overall(1))
        memory_store.backup_overall_memory(overall_path)
        _write(overall_path, _overall(2))

        memory_store.backup_overall_memory(overall_path)

        previous = json.loads(
            (tmp_path / memory_store.PREVIOUS_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert previous == _overall(2)

    def test_全体記憶がまだ無いときは退避しない(self, tmp_path) -> None:
        backup_path = memory_store.backup_overall_memory(
            tmp_path / "overall.json"
        )

        assert backup_path is None
        assert not (tmp_path / memory_store.PREVIOUS_FILENAME).exists()


class TestRollback:
    def test_1回前の記憶を現行へ戻す(self, tmp_path) -> None:
        paths = memory_store.overall_memory_paths(tmp_path, "かまいたちの夜")
        _write(paths.overall, _overall(2))
        _write(paths.previous, _overall(1))

        result = memory_store.rollback_overall_memory(tmp_path, "かまいたちの夜")

        assert result.restored == _overall(1)
        assert result.replaced == _overall(2)
        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(1)

    def test_ロールバックは再実行で取り消せる(self, tmp_path) -> None:
        paths = memory_store.overall_memory_paths(tmp_path, "かまいたちの夜")
        _write(paths.overall, _overall(2))
        _write(paths.previous, _overall(1))

        memory_store.rollback_overall_memory(tmp_path, "かまいたちの夜")
        memory_store.rollback_overall_memory(tmp_path, "かまいたちの夜")

        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(2)
        assert json.loads(paths.previous.read_text(encoding="utf-8")) == _overall(1)

    def test_1回前の記憶が無ければエラーにする(self, tmp_path) -> None:
        paths = memory_store.overall_memory_paths(tmp_path, "かまいたちの夜")
        _write(paths.overall, _overall(2))

        with pytest.raises(memory_store.RollbackError):
            memory_store.rollback_overall_memory(tmp_path, "かまいたちの夜")

        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(2)

    def test_現行の記憶が消えていても1回前から復元する(self, tmp_path) -> None:
        paths = memory_store.overall_memory_paths(tmp_path, "かまいたちの夜")
        _write(paths.previous, _overall(1))

        result = memory_store.rollback_overall_memory(tmp_path, "かまいたちの夜")

        assert result.replaced is None
        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(1)
        assert not paths.previous.exists()

    def test_壊れた1回前の記憶では入れ替えない(self, tmp_path) -> None:
        paths = memory_store.overall_memory_paths(tmp_path, "かまいたちの夜")
        _write(paths.overall, _overall(2))
        paths.previous.write_text("{壊れたJSON", encoding="utf-8")

        with pytest.raises(memory_store.RollbackError):
            memory_store.rollback_overall_memory(tmp_path, "かまいたちの夜")

        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(2)


class TestRollbackCli:
    def test_設定ファイルからタイトルと記憶フォルダを解決する(
        self, tmp_path, capsys
    ) -> None:
        config_path = tmp_path / "game-commentary.toml"
        config_path.write_text(
            'title = "かまいたちの夜"\nmemory_dir = "output/memory"\n',
            encoding="utf-8",
        )
        memory_dir = tmp_path / "output" / "memory"
        paths = memory_store.overall_memory_paths(memory_dir, "かまいたちの夜")
        _write(paths.overall, _overall(2))
        _write(paths.previous, _overall(1))

        exit_code = memory_rollback.main(
            ["--config", str(config_path), "--yes"]
        )

        assert exit_code == 0
        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(1)

    def test_dry_runでは書き換えない(self, tmp_path, capsys) -> None:
        memory_dir = tmp_path / "memory"
        paths = memory_store.overall_memory_paths(memory_dir, "かまいたちの夜")
        _write(paths.overall, _overall(2))
        _write(paths.previous, _overall(1))

        exit_code = memory_rollback.main(
            [
                "--memory-dir",
                str(memory_dir),
                "--title",
                "かまいたちの夜",
                "--dry-run",
            ]
        )

        assert exit_code == 0
        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(2)
        assert json.loads(paths.previous.read_text(encoding="utf-8")) == _overall(1)

    def test_確認プロンプトで拒否すると書き換えない(
        self, tmp_path, monkeypatch
    ) -> None:
        memory_dir = tmp_path / "memory"
        paths = memory_store.overall_memory_paths(memory_dir, "かまいたちの夜")
        _write(paths.overall, _overall(2))
        _write(paths.previous, _overall(1))
        monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

        exit_code = memory_rollback.main(
            ["--memory-dir", str(memory_dir), "--title", "かまいたちの夜"]
        )

        assert exit_code == 1
        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(2)

    def test_1回前の記憶が無ければ終了コード1で知らせる(
        self, tmp_path, capsys
    ) -> None:
        memory_dir = tmp_path / "memory"
        paths = memory_store.overall_memory_paths(memory_dir, "かまいたちの夜")
        _write(paths.overall, _overall(2))

        exit_code = memory_rollback.main(
            [
                "--memory-dir",
                str(memory_dir),
                "--title",
                "かまいたちの夜",
                "--yes",
            ]
        )

        assert exit_code == 1
        assert "1回前" in capsys.readouterr().err


class _StubPlanner:
    """記憶生成のResponses呼び出しを固定JSONで置き換えるスタブ。"""

    def __init__(self) -> None:
        self.phases: list[str] = []

    def generate_text(
        self,
        *,
        phase: str,
        instructions: str,
        use_conversation_history: bool = False,
        planner_instructions: str = "",
    ) -> TextResult:
        self.phases.append(phase)
        if phase == "session_memory":
            payload = {
                "summary": "今回の実況",
                "key_events": [],
                "characters": [],
                "important_choices": [],
                "unresolved_threads": [],
                "commentator_impression": "楽しかった",
                "next_start_point": "次の選択肢",
            }
        else:
            payload = {
                "story_summary": "全体のあらすじ",
                "characters": [],
                "important_choices": [],
                "unresolved_threads": [],
                "current_state": "ペンション",
                "commentator_perspective": "犯人は不明",
                "next_start_point": "次の選択肢",
            }
        return TextResult(
            text=json.dumps(payload, ensure_ascii=False),
            response_id=f"resp-{phase}",
        )


class TestCreateSessionMemoriesWiring:
    def test_全体記憶の更新前に1回前の記憶を退避する(self, tmp_path) -> None:
        memory_dir = tmp_path / "memory"
        paths = memory_store.overall_memory_paths(memory_dir, "かまいたちの夜")
        _write(paths.overall, _overall(1))

        commentary_module._create_session_memories(
            planner=_StubPlanner(),
            root=tmp_path / "run",
            memory_dir=memory_dir,
            title="かまいたちの夜",
            memory_model="gpt-5.6-sol",
            termination_reason="duration_elapsed",
            elapsed_seconds=60.0,
            records=[{"turn": 1, "text": "本文"}],
        )

        assert json.loads(paths.previous.read_text(encoding="utf-8")) == _overall(1)
        updated = json.loads(paths.overall.read_text(encoding="utf-8"))
        assert updated["session_count"] == 2
        assert updated["story_summary"] == "全体のあらすじ"

    def test_全体記憶の生成に失敗したら退避も行わない(self, tmp_path) -> None:
        memory_dir = tmp_path / "memory"
        paths = memory_store.overall_memory_paths(memory_dir, "かまいたちの夜")
        _write(paths.overall, _overall(1))

        class _FailingPlanner(_StubPlanner):
            def generate_text(self, *, phase: str, **kwargs) -> TextResult:
                if phase == "overall_memory":
                    raise RuntimeError("生成失敗")
                return super().generate_text(phase=phase, **kwargs)

        commentary_module._create_session_memories(
            planner=_FailingPlanner(),
            root=tmp_path / "run",
            memory_dir=memory_dir,
            title="かまいたちの夜",
            memory_model="gpt-5.6-sol",
            termination_reason="duration_elapsed",
            elapsed_seconds=60.0,
            records=[{"turn": 1, "text": "本文"}],
        )

        assert not paths.previous.exists()
        assert json.loads(paths.overall.read_text(encoding="utf-8")) == _overall(1)
